import torch
from transformers import LayoutLMForTokenClassification
from peft import get_peft_model, LoraConfig, TaskType

def make_result_dict_light(result_dict):
    """
    When creating the result dictionnary object, we keep track of
    the predictions and labels in order to be able to dive in
    what happened when things go wrong. But keeping track of this
    takes a lot of storage space, and make the pickle files too heavy
    to be pushed on github.
    This functions removes the heaviest keys of a result_dictionnary object.
    [NOTE]: the dictionnary is modified in place.
    """
    KEYS_TO_POP = ['preds_token', 'labels_token', 'preds_word', 'labels_word']
    for k in result_dict.keys():
        if isinstance(result_dict[k], dict):
            make_result_dict_light(result_dict[k])
        else:
            if k in KEYS_TO_POP:
                _ = result_dict.pop(k)


def save_trained_lora_model(model, path):
    """
    Saves only the trained layers from the LoRa model.
    NOTE: this had to be implemented by hand because the
    'save_pretrained' function from Peft did not save the 
    word_embedding layer which was modified during training.
    """
    lora_adapters = {}
    for name, param in model.named_parameters():
        if 'lora' in name or 'classifier' in name or 'word_embeddings' in name:
            lora_adapters[name] = param

    torch.save(lora_adapters, path)


def load_trained_lora_model(id2label, label2id, r, path):
    """
    Loads a pre-trained LoRa model correctly.
    """

    def set_nested_attribute(obj, attr_name, value):
        """
        Set a nested attribute on an object.
        """
        attrs = attr_name.split('.')
        for attr in attrs[:-1]:
            obj = getattr(obj, attr)
        setattr(obj, attrs[-1], value)

    lora_adapters = torch.load(path)

    model = LayoutLMForTokenClassification.from_pretrained("microsoft/layoutlm-base-uncased", num_labels=len(id2label), id2label=id2label, label2id=label2id)
    peft_config = LoraConfig(
        task_type=TaskType.TOKEN_CLS, inference_mode=False, r=r, lora_alpha=0
    )
    model = get_peft_model(model, peft_config)

    for name, param in lora_adapters.items():
        set_nested_attribute(model, name, param)

    return model