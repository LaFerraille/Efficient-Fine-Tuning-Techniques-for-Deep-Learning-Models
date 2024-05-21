import torch
from torch.utils.data import DataLoader
from datasets import load_dataset, Features, Sequence, ClassLabel, Value, Array2D
from torch.utils.data.dataset import Subset
from transformers import LayoutLMTokenizer
import matplotlib.pyplot as plt
import numpy as np
import cv2


plt.style.use("ggplot")
plt.rc("font", family="serif")
plt.rc("legend", fontsize=16)

BLUE = [0, 0, 225]
GREEN = [0, 255, 0]
RED = [255, 0, 0]
YELLOW = [255, 255, 0]
label_to_color = {"question": RED, "answer": GREEN, "header": BLUE, "o": YELLOW}


def encode_sample_for_layoutLM(
    tokenizer: LayoutLMTokenizer, sample, id2label, label2id, max_seq_length=512
):
    """
    Encode one sample (i.e one image) from the funsd Dataset.
    This means:
        - Tokenize the words and adapt their bboxes and labels (a word can yield several tokens)
    """

    token_boxes = []
    aligned_tags = []

    # Tokenize the Words
    for word, box, tag in zip(sample["words"], sample["bboxes"], sample["ner_tags"]):
        label = id2label[tag]
        words_tokens = tokenizer.tokenize(word)
        # NOTE: Maybe we could try refining the bbox handling so that the original bbox
        # is splet among tokens instead of just shared
        token_boxes.extend([box] * len(words_tokens))
        if label != "O":
            label_root = label.split('-')[-1]
            next_tag = label2id['I-' + label_root]
        else:
            next_tag = tag
        aligned_tags.append(tag)
        aligned_tags.extend([next_tag for _ in range(len(words_tokens) - 1)])

    # Clip the token count to the sequence_ma_length
    special_tokens_count = 2
    if len(token_boxes) > max_seq_length - special_tokens_count:
        token_boxes = token_boxes[: (max_seq_length - special_tokens_count)]
        aligned_tags = aligned_tags[: (max_seq_length - special_tokens_count)]

    # Add special tokens
    aligned_tags = [-100] + aligned_tags + [-100]
    token_boxes = [[0, 0, 0, 0]] + token_boxes + [[1000, 1000, 1000, 1000]]

    encoding = tokenizer(
        " ".join(sample["words"]), padding="max_length", truncation=True
    )

    # Add padding to the encoding so that every sample is of size max_seq_length
    pad_token_box = [0, 0, 0, 0]
    padding_length = max_seq_length - len(
        tokenizer(" ".join(sample["words"]), truncation=True)["input_ids"]
    )
    token_boxes += [pad_token_box] * padding_length
    aligned_tags += [-100] * padding_length

    encoding["bboxes"] = token_boxes
    encoding["ner_tags"] = aligned_tags

    del sample["id"]
    del sample["words"]

    return encoding


def prepare_dataloader_for_layoutLM(
    dataset, features, tokenizer, batch_size, id2label, label2id, split: bool = False
) -> DataLoader:
    """
    Returns a DataLoader object, of prepared funsd data for layoutLM
    """

    def _encode_sample_for_layoutLM(sample):
        return encode_sample_for_layoutLM(tokenizer=tokenizer, sample=sample, id2label=id2label, label2id=label2id)

    encoded_data = dataset.map(_encode_sample_for_layoutLM, features=features)
    encoded_data.set_format(type="torch", columns=list(features.keys()))

    if split:
        dataset_length = len(dataset)
        subset_size = int(0.5 * dataset_length)
        subset1_indices = torch.arange(dataset_length)[:subset_size]
        subset2_indices = torch.arange(dataset_length)[subset_size:]
        subset1 = Subset(encoded_data, subset1_indices)
        subset2 = Subset(encoded_data, subset2_indices)
        dataloader1 = DataLoader(subset1, batch_size=batch_size, shuffle=False)
        dataloader2 = DataLoader(subset2, batch_size=batch_size, shuffle=False)
        return dataloader1, dataloader2

    else:
        dataloader = DataLoader(encoded_data, batch_size=batch_size, shuffle=True)
        return dataloader


def load_and_prepare_dataset_for_layoutLM(
    tokenizer: LayoutLMTokenizer = None, batch_size: int = 1
):
    """
    The Dataset is already publicly available but a few transformations
    are needed so that it is ready to use by the layoutLM model.
    Originally, the Dataset is already split between 'train' (149 images)
    and 'test' (50 images) and has the features:
        - id
        - bboxes
        - ner_tags
        - image_path
    Returns:
        train, test and val DataLoaders
    """
    # Retrieve original Dataset
    original_dataset = load_dataset("nielsr/funsd", trust_remote_code=True)
    train_val = original_dataset["train"].train_test_split(test_size=0.3)

    # Get label mapping
    labels = original_dataset["train"].features["ner_tags"].feature.names
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for i, l in enumerate(labels)}

    TRAINING_FEATURES = Features(
        {
            "input_ids": Sequence(Value(dtype="int64")),
            "bboxes": Array2D(dtype="int64", shape=(512, 4)),
            "attention_mask": Sequence(Value(dtype="int64")),
            "token_type_ids": Sequence(Value(dtype="int64")),
            "ner_tags": Sequence(ClassLabel(names=list(id2label.keys()))),
            "image_path": Value(dtype="string"),
        }
    )

    if tokenizer is None:
        tokenizer = LayoutLMTokenizer.from_pretrained("microsoft/layoutlm-base-uncased")

    train_dataloader = prepare_dataloader_for_layoutLM(
        train_val["train"],
        features=TRAINING_FEATURES,
        tokenizer=tokenizer,
        batch_size=batch_size,
        id2label=id2label,
        label2id=label2id,
    )

    val_dataloader = prepare_dataloader_for_layoutLM(
        train_val["test"],
        features=TRAINING_FEATURES,
        tokenizer=tokenizer,
        batch_size=batch_size,
        id2label=id2label,
        label2id=label2id,
    )

    test_dataloader = prepare_dataloader_for_layoutLM(
        original_dataset["test"],
        features=TRAINING_FEATURES,
        tokenizer=tokenizer,
        batch_size=batch_size,
        id2label=id2label,
        label2id=label2id,
    )

    return id2label, label2id, train_dataloader, val_dataloader, test_dataloader


def visualize_sample(sample, id2label: dict, source: str = 'dataset'):
    """
    Shows an image from the dataset where it's writing is color coded based onk it's
    corresponding label.
    Parameters:
        - sample: a sample of the data
        - id2label: a dict that maps an id to a label
        - source: either 'dataset' or 'inference'
    """
    img_path = sample["image_path"][0]
    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    encoded_bboxes = sample["bboxes"][0, :, :].squeeze().tolist()
    if source == 'dataset':
        encoded_tags = sample["ner_tags"][0, :].squeeze().tolist()
    else:
        assert source=='inference', "source parameter must be either 'dataset' or 'inference'"
        encoded_tags = sample["predicted_tags"][0, :].squeeze().tolist()
    bboxes = []
    ner_tags = []

    for i, bb in enumerate(encoded_bboxes):
        if bb not in bboxes and bb not in [[0, 0, 0, 0], [1000, 1000, 1000, 1000]]:
            bboxes.append(bb)
            ner_tags.append(encoded_tags[i])

    # Denormalize the bboxes
    bboxes = [
        [
            int(bb[0] * (w / 1000)),
            int(bb[1] * (h / 1000)),
            int(bb[2] * (w / 1000)),
            int(bb[3] * (h / 1000)),
        ]
        for bb in bboxes
    ]

    for bb, tag in zip(bboxes, ner_tags):
        image_crop = img[bb[1] : bb[3], bb[0] : bb[2], :]
        image_crop[
            np.where((image_crop < [127, 127, 127]).all(axis=2))
        ] = label_to_color[id2label[tag].split("-")[-1].lower()]
        img[bb[1] : bb[3], bb[0] : bb[2], :] = image_crop
        cv2.rectangle(
            img,
            (bb[0], bb[1]),
            (bb[2], bb[3]),
            color=label_to_color[id2label[tag].split("-")[-1].lower()],
        )

    return img
