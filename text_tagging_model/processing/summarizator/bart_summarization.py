from transformers import MBartForConditionalGeneration, MBartTokenizer


class MBartSummarizator:
    def __init__(
        self,
        model_name: str = "IlyaGusev/mbart_ru_sum_gazeta",
        device: str = "cpu",
    ) -> None:
        self.tokenizer = MBartTokenizer.from_pretrained(model_name)
        self.model = MBartForConditionalGeneration.from_pretrained(model_name).to(device)
        self.device = device

    def get_summary(self, text: str) -> str:
        input_ids = self.tokenizer(
            [text],
            add_special_tokens=True,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )["input_ids"]

        output_ids = self.model.generate(
            input_ids=input_ids.to(self.device),
            max_length=128,
            no_repeat_ngram_size=3,
            num_beams=10,
        )[0]

        if self.device == "cpu":
            output_ids.detach().cpu().numpy()

        summary: str = self.tokenizer.decode(output_ids, skip_special_tokens=True)

        return summary
