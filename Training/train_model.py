import torch
import sys
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
import pandas as pd
import numpy as np
from collections import Counter
import wandb
import time

from peft import get_peft_model, LoraConfig, LoHaConfig, LoKrConfig, AdaLoraConfig, TaskType 

sys.path.append('..')
from utils.utils import save_trained_lora_model, load_trained_lora_model


def one_epoch_step(
    model,
    optimizer: AdamW,
    device: torch.device,
    dataloader: DataLoader,
    id2label: dict,
    epoch_type: str = "Training",
):
    epoch_mean_loss = 0
    total_preds = None
    total_bboxes = None
    total_labels = None

    if epoch_type == "Training":
        model.train()
    else:
        model.eval()

    for batch in tqdm(dataloader, desc=epoch_type):
        # Compute predictions
        labels = batch["ner_tags"].to(device)
        outputs = model(
            input_ids=batch["input_ids"].to(device),
            bbox=batch["bboxes"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            token_type_ids=batch["token_type_ids"].to(device),
            labels=labels,
        )
        bboxes = batch["bboxes"]
        logits = outputs.logits
        new_preds_token = logits.detach().cpu().numpy()
        new_preds_token = np.argmax(new_preds_token, axis=2)

        loss = outputs.loss
        epoch_mean_loss += loss.item()

        if total_preds is not None:
            total_preds = np.append(total_preds, new_preds_token, axis=0)
            total_bboxes = np.append(total_bboxes, bboxes.cpu().numpy(), axis=0)
            total_labels = np.append(total_labels, labels.cpu().numpy(), axis=0)
        else:
            total_preds = new_preds_token
            total_bboxes = bboxes.cpu().numpy()
            total_labels = labels.cpu().numpy()

        # Backprop if it is a training epoch
        if epoch_type == "Training":
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    epoch_mean_loss /= len(dataloader)*dataloader.batch_size
    # Evaluate the model

    evaluation_dict = evaluate_preds(
        preds_token=total_preds,
        bboxes=total_bboxes,
        labels_token=total_labels,
        original_id2label=id2label,
    )
    f1 = evaluation_dict['f1']
    acc = evaluation_dict['precision']

    print(f"Epoch {epoch_type} loss: {epoch_mean_loss}, {epoch_type} F1 Score: {f1}, {epoch_type} Accuracy: {acc}")

    return epoch_mean_loss, f1, acc

def apply_peft(
        model,
        peft_method : LoraConfig | LoHaConfig | LoKrConfig | AdaLoraConfig,
        r : int = 8,
        target_modules : list = None
    ):
    """
    Apply the PEFT algorithm to the model
    Parameters:
        - model: model to be trained
        - r: number of bits to quantize the weights
        - target_modules: list of modules to quantize

    Return:
        The model with the PEFT algorithm applied
    """

    # Apply the PEFT algorithm to the model
    if peft_method.__name__ == "LoraConfig" or peft_method.__name__ == "AdaLoraConfig":
        peft_config = peft_method(task_type=TaskType.TOKEN_CLS, r=r, lora_alpha=8*r, target_modules=target_modules)
    else:
        peft_config = peft_method(task_type=TaskType.TOKEN_CLS, r=r, alpha=8*r, target_modules=target_modules)
    model = get_peft_model(model, peft_config)

    print(model.print_trainable_parameters())

    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return model, trainable_parameters


def fine_tune_model(
    model,
    device: torch.device,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    id2label: dict,
    optimizer: torch.optim.AdamW | None = None,
    scheduler: torch.optim.lr_scheduler.LambdaLR | None = None,
    num_epoch_max: int = 10,
    lr: float = 5e-5,
    early_stopping: bool = True,
    test_frequency: int = 1,
    model_folder: str = 'Full_Fine_Tuning',
    model_name: str | None = None,
    wandb_project_name: str = 'Full_Fine_Tuning'
):
    """
    Realize a full fine tuning of the model (every trainable weight is updated)
    Parameters:
        - model: model to be trained
        - device: device on which to train the model
        - train_dataloader: Dataloader with the training samples
        - val_dataloader: Dataloader with the validation samples
        - num_epoch_max: maximum number of epoch to do
        - lr: learning rate
        - early_stopping: Wheter or not we want to stop early if our model
            does not get better on the validation set. Current criterion is if it is
            not better on the validation set 2 epochs in a row we stop the training.
        - test_frequency: number of epochs between each pass on the val dataloader.
        - model_folder: Folder in which to save the folder. It needs to be created beforehand.
        - model_name: Adds a name to the model so that is saved in another file than
            the default one. Use a name descriptive of the experiment like
            'lora_r_32_alpha_8'
    Return:
        Nothing, the model weights are modified in place.
    """

    # Initialize everything
    model.to(device)

    training_losses = []
    val_losses = []

    training_accs = []
    val_accs = []

    training_f1s = []
    val_f1s = []

    best_val_loss = None
    consecutive_non_amelioration = 0
    max_memory_per_epoch = []
    if optimizer is None:
        optimizer = AdamW(model.parameters(), lr=lr)
    name_extension = "" if model_name is None else f"_{model_name}"
    is_peft_model = str(type(model)).split('.')[0].endswith('peft')

    wandb.init(project="LoRA Project", name=wandb_project_name)

    # Training loop
    print("Starting Training")
    for epoch in range(num_epoch_max):

        t0 = time.perf_counter()

        if torch.cuda.is_available() : 
            torch.cuda.reset_peak_memory_stats(device=None)
        else :
            pass
        # On the training dataloader
        print(f"Epoch: {epoch + 1}/{num_epoch_max}")
        training_loss, training_f1, training_acc = one_epoch_step(
                model=model,
                optimizer=optimizer,
                device=device,
                dataloader=train_dataloader,
                id2label=id2label,
                epoch_type="Training",
            )
        
        batch_size = train_dataloader.batch_size
        delta_t = time.perf_counter() - t0
        wandb.log({"Time elapsed": delta_t, "it_s": batch_size/delta_t})
        
        training_losses.append(training_loss)
        training_f1s.append(training_f1)
        training_accs.append(training_acc)


        if scheduler is not None:
            scheduler.step()

        if is_peft_model:
            save_trained_lora_model(model, f"../Saved_Models/{model_folder}/Best" + name_extension  + ".pth")
        else:
            torch.save(
                model.state_dict(),
                f"../Saved_Models/{model_folder}/Last" + name_extension + ".pth",
            )

        # On the val dataloader
        if epoch % test_frequency == 0:
            with torch.no_grad():
                new_val_loss, val_f1, val_acc = one_epoch_step(
                        model=model,
                        optimizer=optimizer,
                        device=device,
                        dataloader=val_dataloader,
                        id2label=id2label,
                        epoch_type="Test",
                    )
                
                val_losses.append(new_val_loss)
                val_f1s.append(val_f1)
                val_accs.append(val_acc)

            if best_val_loss is None or new_val_loss < best_val_loss:
                consecutive_non_amelioration = 0
                best_val_loss = new_val_loss
                if is_peft_model:
                    save_trained_lora_model(model, f"../Saved_Models/{model_folder}/Best" + name_extension  + ".pth")
                else:
                    torch.save(
                        model.state_dict(),
                        f"../Saved_Models/{model_folder}/Best" + name_extension + ".pth",
                    )
            else:
                consecutive_non_amelioration += 1

            if early_stopping and consecutive_non_amelioration == 2:
                print("Terminated training early due to no improvements")
                break
            
    if torch.cuda.is_available(): 
        memory_usage = torch.cuda.max_memory_allocated(device=device)/1000**3
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available() :
        memory_usage = torch.mps.driver_allocated_memory()/1000**3
        torch.mps.empty_cache()
    else :
        pass

    wandb.summary['memory_GB'] = memory_usage
    wandb.finish()

    
    print("Training finished.")

    return training_losses, val_losses, training_f1s, val_f1s, training_accs, val_accs

def most_frequent(List):
    """
    Returns the most frequent element in a list
    """
    occurence_count = Counter(List)
    return occurence_count.most_common(1)[0][0]




def evaluate_model(
    model, device: torch.device, test_dataloader: DataLoader, original_id2label: dict, include_other: bool = False
):
    """
    Computes different types of metrics such as the accuracy, recall and f1_score
    obtained by the model on the test Dataset.
    """

    model.to(device)

    pad_token_label_id = -100
    id2label = original_id2label.copy()
    id2label[pad_token_label_id] = str(pad_token_label_id)

    eval_loss = 0.0
    preds_token = None
    preds_word = None
    out_label_token = None
    out_label_word = None

    model.eval()
    for batch in tqdm(test_dataloader, desc="Evaluating"):
        with torch.no_grad():
            labels = batch["ner_tags"].to(device)
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                bbox=batch["bboxes"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                token_type_ids=batch["token_type_ids"].to(device),
                labels=labels,
            )

            batch_eval_loss = outputs.loss
            logits = outputs.logits

            eval_loss += batch_eval_loss.item()
            new_preds_token = logits.detach().cpu().numpy()
            new_preds_token = np.argmax(new_preds_token, axis=2)
            new_preds_token = np.array([[id2label[tag] for tag in row] for row in new_preds_token])
            new_preds_word = []
            new_out_label_word = []
            for i in range(new_preds_token.shape[0]):
                new_preds_word.append([['100']])
                new_out_label_word.append(['100'])
                previous_bb = torch.Tensor([0, 0, 0, 0])
                for j in range(1, new_preds_token.shape[1]):
                    bb = batch["bboxes"][i, j]
                    if (bb == previous_bb).all():
                        new_preds_word[-1][-1].append(new_preds_token[i, j].split('-')[-1])
                    else:
                        previous_bb = bb
                        new_preds_word[-1].append([new_preds_token[i, j].split('-')[-1]])
                        new_out_label_word[-1].append(id2label[labels.detach().cpu().numpy()[i, j]].split('-')[-1])

            # A word contains the prediction drom each of it's token now let's keep only the most frequent one.
            # Apply the most_frequent function to each sublist
            new_preds_word = [[most_frequent(inner_list) for inner_list in outer_list] for outer_list in new_preds_word]

            # compute the predictions
            if preds_token is None:
                preds_token = new_preds_token
                out_label_token = labels.detach().cpu().numpy()
                preds_word = new_preds_word
                out_label_word = new_out_label_word
            else:
                preds_token = np.append(preds_token, new_preds_token, axis=0)
                out_label_token = np.append(
                    out_label_token, labels.detach().cpu().numpy(), axis=0
                )
                preds_word.extend(new_preds_word)
                out_label_word.extend(new_out_label_word)

    # compute average evaluation loss
    eval_loss /= len(test_dataloader)*test_dataloader.batch_size

    out_label_list_token = [[] for _ in range(out_label_token.shape[0])]
    preds_list_token = [[] for _ in range(out_label_token.shape[0])]

    # Computes prediction at token level
    for i in range(out_label_token.shape[0]):
        for j in range(out_label_token.shape[1]):
            if out_label_token[i, j] != pad_token_label_id:
                if not include_other and (id2label[out_label_token[i, j]] == "O" and preds_token[i, j] == "O"):
                    pass
                else:
                    out_label_list_token[i].append(id2label[out_label_token[i, j]])
                    preds_list_token[i].append(preds_token[i, j])

    out_label_list_word = [[] for _ in range(len(out_label_word))]
    preds_list_word = [[] for _ in range(len(out_label_word))]

    # Computes prediction at bbox level
    for i in range(len(out_label_word)):
        for j in range(len(out_label_word[i])):
                if out_label_word[i][j] != str(pad_token_label_id).split('-')[-1]:
                    if not include_other and (out_label_word[i][j] == "O" and preds_word[i][j] == "O"):
                        pass
                    else:
                        out_label_list_word[i].append(out_label_word[i][j])
                        preds_list_word[i].append(preds_word[i][j])

    results = {
        "loss": eval_loss,
        "preds_token": preds_list_token,
        "labels_token": out_label_list_token,
        "preds_word": preds_list_word,
        "labels_word": out_label_list_word,
    }

    y_pred_detailed, y_pred_relaxed = [], []
    y_true_detailed, y_true_relaxed = [], []
    detailed_classes = [
        "B-HEADER",
        "I-HEADER",
        "B-QUESTION",
        "I-QUESTION",
        "B-ANSWER",
        "I-ANSWER",
    ]
    if include_other:
        detailed_classes.append("O")

    relaxed_classes = ["HEADER", "QUESTION", "ANSWER"]
    if include_other:
        relaxed_classes.append("O")

    # Prediction are image per image (i.e array of shape[nb_sample, tokens_per_sample])
    # we flatten everything to compute metrics
    for predicted_tokens, token_labels in zip(preds_list_token, out_label_list_token):
        y_pred_detailed.extend(list(predicted_tokens))
        y_true_detailed.extend(list(token_labels))

    for predicted_words, word_labels in zip(preds_list_word, out_label_list_word):
        y_pred_relaxed.extend([l for l in predicted_words])
        y_true_relaxed.extend([l for l in word_labels])

    # Build confusion matrices
    cf_matrix_detailed = confusion_matrix(
        y_true_detailed, y_pred_detailed, labels=detailed_classes
    )
    cf_matrix_detailed_df = pd.DataFrame(
        cf_matrix_detailed / np.sum(cf_matrix_detailed, axis=1)[:, None],
        index=detailed_classes,
        columns=detailed_classes,
    )
    cf_matrix_relaxed = confusion_matrix(
        y_true_relaxed, y_pred_relaxed, labels=relaxed_classes
    )
    cf_matrix_relaxed_df = pd.DataFrame(
        cf_matrix_relaxed / np.sum(cf_matrix_relaxed, axis=1)[:, None],
        index=relaxed_classes,
        columns=relaxed_classes,
    )
    results["conf_matrix_tokens"] = cf_matrix_detailed_df
    results["conf_matrix_words"] = cf_matrix_relaxed_df

    # Get metrics
    detailed_metrics = np.array(
        [
            precision_score(
                y_true=y_true_detailed,
                y_pred=y_pred_detailed,
                labels=detailed_classes,
                average=None,
            ),
            recall_score(
                y_true=y_true_detailed,
                y_pred=y_pred_detailed,
                labels=detailed_classes,
                average=None,
            ),
            f1_score(
                y_true=y_true_detailed,
                y_pred=y_pred_detailed,
                labels=detailed_classes,
                average=None,
            ),
        ]
    )
    detailed_metrics = np.transpose(detailed_metrics)
    detailed_metrics_df = pd.DataFrame(
        data=detailed_metrics,
        index=detailed_classes,
        columns=["Precision", "Recall", "F1"],
    )
    relaxed_metrics = np.array(
        [
            precision_score(
                y_true=y_true_relaxed,
                y_pred=y_pred_relaxed,
                labels=relaxed_classes,
                average=None,
            ),
            recall_score(
                y_true=y_true_relaxed,
                y_pred=y_pred_relaxed,
                labels=relaxed_classes,
                average=None,
            ),
            f1_score(
                y_true=y_true_relaxed,
                y_pred=y_pred_relaxed,
                labels=relaxed_classes,
                average=None,
            ),
        ]
    )
    relaxed_metrics = np.transpose(relaxed_metrics)
    relaxed_metrics_df = pd.DataFrame(
        data=relaxed_metrics,
        index=relaxed_classes,
        columns=["Precision", "Recall", "F1"],
    )
    results["metrics_tokens"] = detailed_metrics_df
    results["metrics_words"] = relaxed_metrics_df
    results['precision'] = precision_score(
        y_true=y_true_relaxed,
        y_pred=y_pred_relaxed,
        labels=relaxed_classes,
        average='micro',
    )
    results['recall'] = recall_score(
        y_true=y_true_relaxed,
        y_pred=y_pred_relaxed,
        labels=relaxed_classes,
        average='micro',
    )
    results['f1'] = f1_score(
        y_true=y_true_relaxed,
        y_pred=y_pred_relaxed,
        labels=relaxed_classes,
        average='micro',
    )

    return results


def evaluate_preds(
    preds_token, bboxes, labels_token, original_id2label: dict, include_other: bool = False
):
    """
    Computes different types of metrics such as the accuracy, recall and f1_score
    obtained by the model on the test Dataset.
    """
    pad_token_label_id = -100
    id2label = original_id2label.copy()
    id2label[pad_token_label_id] = str(pad_token_label_id)

    preds_word = None
    out_label_word = None

    # preds is of size (num elements * 512)
    preds_token = np.array([[id2label[tag] for tag in row] for row in preds_token])
    preds_word = []
    out_label_word = []
    # for 1 doc
    for i in range(preds_token.shape[0]):
        new_preds_word = [['100']]
        new_out_label_word = ['100']
        previous_bb = np.array([0, 0, 0, 0])
        for j in range(1, preds_token.shape[1]):
            bb = bboxes[i, j]
            if (bb == previous_bb).all():
                new_preds_word[-1].append(preds_token[i, j].split('-')[-1])
            else:
                previous_bb = bb
                new_preds_word.append([preds_token[i, j].split('-')[-1]])
                new_out_label_word.append(id2label[labels_token[i, j]].split('-')[-1])

        # A word contains the prediction of each of it's token now let's keep only the most frequent one.
        # Apply the most_frequent function to each sublist
        new_preds_word = [most_frequent(inner_list) for inner_list in new_preds_word]

        preds_word.extend([new_preds_word])
        out_label_word.extend([new_out_label_word])

    out_label_list_token = [[] for _ in range(labels_token.shape[0])]
    preds_list_token = [[] for _ in range(labels_token.shape[0])]

    # Computes prediction at token level
    for i in range(labels_token.shape[0]):
        for j in range(labels_token.shape[1]):
            if labels_token[i, j] != pad_token_label_id:
                if not include_other and (id2label[labels_token[i, j]] == "O" and preds_token[i, j] == "O"):
                    pass
                else:
                    out_label_list_token[i].append(id2label[labels_token[i, j]])
                    preds_list_token[i].append(preds_token[i, j])

    out_label_list_word = [[] for _ in range(len(out_label_word))]
    preds_list_word = [[] for _ in range(len(out_label_word))]

    # Computes prediction at bbox level
    for i in range(len(out_label_word)):
        for j in range(len(out_label_word[i])):
                if out_label_word[i][j] != str(pad_token_label_id).split('-')[-1]:
                    if not include_other and (out_label_word[i][j] == "O" and preds_word[i][j] == "O"):
                        pass
                    else:
                        out_label_list_word[i].append(out_label_word[i][j])
                        preds_list_word[i].append(preds_word[i][j])

    results = {
        "preds_token": preds_list_token,
        "labels_token": out_label_list_token,
        "preds_word": preds_list_word,
        "labels_word": out_label_list_word,
    }

    y_pred_detailed, y_pred_relaxed = [], []
    y_true_detailed, y_true_relaxed = [], []
    detailed_classes = [
        "B-HEADER",
        "I-HEADER",
        "B-QUESTION",
        "I-QUESTION",
        "B-ANSWER",
        "I-ANSWER",
    ]
    if include_other:
        detailed_classes.append("O")

    relaxed_classes = ["HEADER", "QUESTION", "ANSWER"]
    if include_other:
        relaxed_classes.append("O")

    # Prediction are image per image (i.e array of shape[nb_sample, tokens_per_sample])
    # we flatten everything to compute metrics
    for predicted_tokens, token_labels in zip(preds_list_token, out_label_list_token):
        y_pred_detailed.extend(list(predicted_tokens))
        y_true_detailed.extend(list(token_labels))

    for predicted_words, word_labels in zip(preds_list_word, out_label_list_word):
        y_pred_relaxed.extend([l for l in predicted_words])
        y_true_relaxed.extend([l for l in word_labels])

    # Build confusion matrices
    cf_matrix_detailed = confusion_matrix(
        y_true_detailed, y_pred_detailed, labels=detailed_classes
    )
    cf_matrix_detailed_df = pd.DataFrame(
        cf_matrix_detailed / np.sum(cf_matrix_detailed, axis=1)[:, None],
        index=detailed_classes,
        columns=detailed_classes,
    )
    cf_matrix_relaxed = confusion_matrix(
        y_true_relaxed, y_pred_relaxed, labels=relaxed_classes
    )
    cf_matrix_relaxed_df = pd.DataFrame(
        cf_matrix_relaxed / np.sum(cf_matrix_relaxed, axis=1)[:, None],
        index=relaxed_classes,
        columns=relaxed_classes,
    )
    results["conf_matrix_tokens"] = cf_matrix_detailed_df
    results["conf_matrix_words"] = cf_matrix_relaxed_df

    # Get metrics
    detailed_metrics = np.array(
        [
            precision_score(
                y_true=y_true_detailed,
                y_pred=y_pred_detailed,
                labels=detailed_classes,
                average=None,
            ),
            recall_score(
                y_true=y_true_detailed,
                y_pred=y_pred_detailed,
                labels=detailed_classes,
                average=None,
            ),
            f1_score(
                y_true=y_true_detailed,
                y_pred=y_pred_detailed,
                labels=detailed_classes,
                average=None,
            ),
        ]
    )
    detailed_metrics = np.transpose(detailed_metrics)
    detailed_metrics_df = pd.DataFrame(
        data=detailed_metrics,
        index=detailed_classes,
        columns=["Precision", "Recall", "F1"],
    )
    relaxed_metrics = np.array(
        [
            precision_score(
                y_true=y_true_relaxed,
                y_pred=y_pred_relaxed,
                labels=relaxed_classes,
                average=None,
            ),
            recall_score(
                y_true=y_true_relaxed,
                y_pred=y_pred_relaxed,
                labels=relaxed_classes,
                average=None,
            ),
            f1_score(
                y_true=y_true_relaxed,
                y_pred=y_pred_relaxed,
                labels=relaxed_classes,
                average=None,
            ),
        ]
    )
    relaxed_metrics = np.transpose(relaxed_metrics)
    relaxed_metrics_df = pd.DataFrame(
        data=relaxed_metrics,
        index=relaxed_classes,
        columns=["Precision", "Recall", "F1"],
    )
    results["metrics_tokens"] = detailed_metrics_df
    results["metrics_words"] = relaxed_metrics_df
    results['precision'] = precision_score(
        y_true=y_true_relaxed,
        y_pred=y_pred_relaxed,
        labels=relaxed_classes,
        average='micro',
    )
    results['recall'] = recall_score(
        y_true=y_true_relaxed,
        y_pred=y_pred_relaxed,
        labels=relaxed_classes,
        average='micro',
    )
    results['f1'] = f1_score(
        y_true=y_true_relaxed,
        y_pred=y_pred_relaxed,
        labels=relaxed_classes,
        average='micro',
    )

    return results