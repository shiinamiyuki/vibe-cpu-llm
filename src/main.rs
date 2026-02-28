/// CLI entry point for text generation with the Cohere2 (tiny-aya) model.
///
/// Usage:
///   cargo run --bin generate -- --model-dir models/tiny-aya-global --prompt "Hello"
///
/// The generator loads the model and tokenizer, applies the Cohere2 chat
/// template to the prompt, then decodes token-by-token using greedy sampling
/// until it hits <|END_OF_TURN_TOKEN|> or reaches the max token count.
///
/// # Assumptions
/// - The model directory contains `config.json`, `tokenizer.json`,
///   `model.safetensors.index.json`, and the shard files.
/// - Generation uses greedy decoding (argmax) for simplicity.

use std::io::Write;

use tokenizers::Tokenizer;

use vibe_cpu_llm::model::forward::Cohere2Model;

/// End-of-turn token ID used by Cohere2 models as the generation stop signal.
const END_OF_TURN_TOKEN_ID: u32 = 6;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let mut model_dir = "models/tiny-aya-global".to_string();
    let mut prompt = "Hello, how are you?".to_string();
    let mut max_tokens: usize = 128;

    // Simple argument parsing
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model-dir" => {
                i += 1;
                model_dir = args[i].clone();
            }
            "--prompt" => {
                i += 1;
                prompt = args[i].clone();
            }
            "--max-tokens" => {
                i += 1;
                max_tokens = args[i].parse().expect("invalid --max-tokens");
            }
            other => {
                eprintln!("Unknown argument: {}", other);
                std::process::exit(1);
            }
        }
        i += 1;
    }

    // Load tokenizer
    eprintln!("Loading tokenizer from {}/tokenizer.json ...", model_dir);
    let tokenizer_path = format!("{}/tokenizer.json", model_dir);
    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .unwrap_or_else(|e| panic!("Failed to load tokenizer: {}", e));

    // Load model
    let mut model = Cohere2Model::load(&model_dir);

    // Build the chat-formatted prompt using Cohere2's turn token structure:
    //   <BOS_TOKEN><|START_OF_TURN_TOKEN|><|USER_TOKEN|>{prompt}<|END_OF_TURN_TOKEN|>
    //   <|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>
    let chat_prompt = format!(
        "<BOS_TOKEN><|START_OF_TURN_TOKEN|><|USER_TOKEN|>{}<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>",
        prompt
    );

    eprintln!("Prompt: {}", chat_prompt);

    // Tokenize
    let encoding = tokenizer
        .encode(chat_prompt.as_str(), false)
        .unwrap_or_else(|e| panic!("Tokenization failed: {}", e));
    let input_ids: Vec<u32> = encoding.get_ids().to_vec();
    eprintln!("Input tokens ({}): {:?}", input_ids.len(), input_ids);

    // Prefill: run all prompt tokens through the model
    eprintln!("Running prefill ({} tokens)...", input_ids.len());
    let mut logits = Vec::new();
    for (pos, &token_id) in input_ids.iter().enumerate() {
        logits = model.forward(token_id, pos);
        if pos % 10 == 0 {
            eprint!("  pos {} / {} ...\r", pos, input_ids.len());
        }
    }
    eprintln!("Prefill done.                    ");

    // Greedy decode
    eprintln!("Generating (max {} tokens)...", max_tokens);
    let mut generated_tokens: Vec<u32> = Vec::new();
    let mut pos = input_ids.len();

    for step in 0..max_tokens {
        // Argmax over logits
        let next_token = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx as u32)
            .unwrap();

        if next_token == END_OF_TURN_TOKEN_ID {
            eprintln!("\n[END_OF_TURN at step {}]", step);
            break;
        }

        generated_tokens.push(next_token);

        // Decode and print this token immediately
        let token_str = tokenizer
            .decode(&[next_token], false)
            .unwrap_or_else(|_| "�".to_string());
        print!("{}", token_str);
        std::io::stdout().flush().ok();

        // Forward the new token
        logits = model.forward(next_token, pos);
        pos += 1;
    }

    println!();
    eprintln!(
        "Generated {} tokens total.",
        generated_tokens.len()
    );
}
