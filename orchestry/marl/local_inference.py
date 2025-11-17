"""
Local LLM inference with LoRA fine-tuning for MAGRPO.

This module provides wrappers for running local language models
(like Qwen2.5-Coder) with LoRA adapters for parameter-efficient fine-tuning.
"""

import logging

import torch  # type: ignore[import-not-found]
from peft import LoraConfig, PeftModel, get_peft_model  # type: ignore[import-not-found]
from torch.nn import functional  # type: ignore[import-not-found]
from transformers import (  # type: ignore[import-not-found]
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

logger = logging.getLogger(__name__)


class LocalLLMAgent:
    """
    Local LLM agent with LoRA fine-tuning.

    Wraps a Hugging Face causal LM with LoRA adapters for efficient training.
    Supports 4-bit quantization to reduce memory usage.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-Coder-1.5B",
        lora_config: LoraConfig | None = None,
        load_in_4bit: bool = True,
        device: str = "auto",
    ) -> None:
        """
        Initialize local LLM agent.

        Args:
            model_name: HuggingFace model identifier
            lora_config: LoRA configuration (if None, uses default)
            load_in_4bit: Use 4-bit quantization (recommended for memory)
            device: Device placement ("auto", "cuda", "cpu")

        """
        self.model_name = model_name
        self.device = device

        logger.info(f"Loading model: {model_name}")

        # Configure 4-bit quantization
        if load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        else:
            bnb_config = None

        # Load base model
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map=device,
            torch_dtype=torch.float16 if load_in_4bit else torch.float32,
            trust_remote_code=True,
        )

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )

        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Apply LoRA
        if lora_config is None:
            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
            )

        self.model = get_peft_model(self.base_model, lora_config)
        self.model.print_trainable_parameters()

        logger.info(f"Model loaded successfully on {self.model.device}")

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.8,
        top_p: float = 0.95,
        do_sample: bool = True,
    ) -> str:
        """
        Generate a single response.

        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            do_sample: Use sampling (vs greedy)

        Returns:
            generated_text: Generated text (prompt + completion)

        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        return str(self.tokenizer.decode(outputs[0], skip_special_tokens=True))

    def generate_group(
        self,
        prompt: str,
        k: int = 4,
        max_new_tokens: int = 512,
        temperature: float = 0.8,
        top_p: float = 0.95,
    ) -> list[str]:
        """
        Generate k samples in parallel (efficient batching).

        Args:
            prompt: Input prompt
            k: Number of samples to generate
            max_new_tokens: Maximum tokens per sample
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter

        Returns:
            samples: List of k generated texts

        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                num_return_sequences=k,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        return [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

    def compute_log_prob(self, action: str, observation: str) -> torch.Tensor:
        """
        Compute log π(action|observation) for policy gradient.

        Args:
            action: Generated action (code snippet)
            observation: Input observation (prompt)

        Returns:
            log_prob: Log probability of action given observation

        """
        # Concatenate observation and action
        full_text = observation + action

        # Tokenize
        inputs = self.tokenizer(full_text, return_tensors="pt").to(self.model.device)
        obs_inputs = self.tokenizer(observation, return_tensors="pt").to(self.model.device)

        # Get model outputs
        with torch.enable_grad():  # Need gradients for training
            outputs = self.model(**inputs)
            logits = outputs.logits

        # Get action tokens (everything after observation)
        obs_len = obs_inputs.input_ids.shape[1]
        action_tokens = inputs.input_ids[0, obs_len:]

        # Compute log probabilities
        log_probs = functional.log_softmax(logits, dim=-1)

        # Sum log probs over action sequence
        total_log_prob = 0.0
        for i, token in enumerate(action_tokens):
            if obs_len + i < log_probs.shape[1]:
                total_log_prob += log_probs[0, obs_len + i - 1, token]

        return total_log_prob

    def save_lora_weights(self, path: str) -> None:
        """Save only the LoRA adapter weights."""
        self.model.save_pretrained(path)
        logger.info(f"LoRA weights saved to {path}")

    def load_lora_weights(self, path: str) -> None:
        """Load LoRA adapter weights."""
        self.model = PeftModel.from_pretrained(self.base_model, path)
        logger.info(f"LoRA weights loaded from {path}")

    def get_trainable_parameters(self) -> int:
        """Get count of trainable parameters."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def get_model_size_mb(self) -> float:
        """Get model size in MB."""
        param_size = sum(p.numel() * p.element_size() for p in self.model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.model.buffers())
        return float((param_size + buffer_size) / (1024**2))


def create_agent_pair(
    model_name: str = "Qwen/Qwen2.5-Coder-1.5B",
    load_in_4bit: bool = True,
) -> tuple[LocalLLMAgent, LocalLLMAgent]:
    """
    Create a pair of agents for helper/main collaboration.

    Both agents share the same base model but have separate LoRA adapters.

    Args:
        model_name: HuggingFace model identifier
        load_in_4bit: Use 4-bit quantization

    Returns:
        (agent_helper, agent_main): Tuple of two agents

    """
    logger.info("Creating agent pair...")

    # Create first agent (helper)
    agent_helper = LocalLLMAgent(
        model_name=model_name,
        load_in_4bit=load_in_4bit,
    )

    # Create second agent (main)
    agent_main = LocalLLMAgent(
        model_name=model_name,
        load_in_4bit=load_in_4bit,
    )

    logger.info("Agent pair created successfully")
    return agent_helper, agent_main
