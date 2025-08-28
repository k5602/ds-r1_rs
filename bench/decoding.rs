use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use ds_r1_rs::inference::generation::{GenerationCache, GenerationConfig, TextGenerator};
use ds_r1_rs::inference::sampling::SamplingConfig;
use ds_r1_rs::model::config::ModelConfig;
use ds_r1_rs::model::transformer::DeepSeekR1Model;
use ds_r1_rs::utils::tokenizer::{Tokenizer, TokenizerConfig};

fn make_prompt(len: usize) -> String {
    // Deterministic prompt of given length to avoid randomness overhead in benches
    // Uses a simple repeating alphabet pattern.
    const ALPHABET: &[u8] = b"abcdefghijklmnopqrstuvwxyz ";
    let mut s = String::with_capacity(len);
    for i in 0..len {
        let ch = ALPHABET[i % ALPHABET.len()] as char;
        s.push(ch);
    }
    s
}

fn bench_decoding(c: &mut Criterion) {
    // Compare decoding throughput (tokens/sec) without cache vs with cache
    // using varying prompt lengths.
    let prompt_lens = [64usize, 128, 256];
    let gen_tokens = 128usize;

    let mut group = c.benchmark_group("decoding_tokens_per_sec");
    group.sample_size(10);
    group.warm_up_time(std::time::Duration::from_millis(500));
    group.measurement_time(std::time::Duration::from_secs(10));

    for &prompt_len in &prompt_lens {
        // Shared components per prompt length to avoid measuring init cost.
        let model_config = ModelConfig::default();
        let mut model = DeepSeekR1Model::new(model_config).expect("model init failed");

        let tok_cfg = TokenizerConfig::default();
        let tokenizer = Tokenizer::new(tok_cfg).expect("tokenizer init failed");

        let sampling_cfg = SamplingConfig::default();
        let mut generator = TextGenerator::new(sampling_cfg);

        let gen_cfg = GenerationConfig {
            max_tokens: gen_tokens,
            temperature: 0.0, // greedy for determinism in benches
            top_k: None,
            top_p: None,
            stop_tokens: vec![],
            repetition_penalty: 1.0,
        };

        let prompt = make_prompt(prompt_len);

        // Use throughput based on number of tokens generated per iteration.
        group.throughput(Throughput::Elements(gen_tokens as u64));

        // Baseline: no cache (recomputes full prefix each step)
        group.bench_with_input(BenchmarkId::new("no_cache", prompt_len), &prompt, |b, p| {
            b.iter(|| {
                let output = generator
                    .generate(
                        black_box(&mut model),
                        black_box(&tokenizer),
                        black_box(p),
                        black_box(&gen_cfg),
                    )
                    .expect("generation (no_cache) failed");
                black_box(output.tokens_generated);
            });
        });

        // With cache: true incremental decoding
        group.bench_with_input(
            BenchmarkId::new("with_cache", prompt_len),
            &prompt,
            |b, p| {
                b.iter(|| {
                    let mut cache = GenerationCache::new();
                    let output = generator
                        .generate_with_cache(
                            black_box(&mut model),
                            black_box(&tokenizer),
                            black_box(p),
                            black_box(&gen_cfg),
                            black_box(&mut cache),
                        )
                        .expect("generation (with_cache) failed");
                    black_box(output.tokens_generated);
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_decoding);
criterion_main!(benches);
