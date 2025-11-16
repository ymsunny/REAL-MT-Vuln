# REAL-MT-Vuln
\textbf{RE}trieval-\textbf{A}ugmented \textbf{L}LM-based \textbf{M}achine \textbf{T}ranslation (REAL-MT) shows promise for knowledge-intensive tasks like idiomatic translation, but its reliability under noisy retrieval, a common challenge in real-world deployment, remains poorly understood. To address this gap, we propose a noise synthesis framework and new metrics to systematically evaluate REAL-MT’s reliability across high-, medium-, and low-resource language pairs. Using both open- and closed-sourced models, including standard LLMs and large reasoning models (LRMs), we find that models heavily rely on retrieved context, and this dependence is significantly more detrimental in low-resource language pairs, producing nonsensical translations. Although LRMs possess enhanced reasoning capabilities, they show no improvement in error correction and are even more susceptible to noise, tending to rationalize incorrect contexts. Attention analysis reveals a shift from the source idiom to noisy content, while confidence increases despite declining accuracy, indicating poor self-monitoring. To mitigate these issues, we investigate training-free and fine-tuning strategies, which improve robustness at the cost of performance in clean contexts, revealing a fundamental trade-off. Our findings highlight the limitations of current approaches, underscoring the need for self-verifying integration mechanisms.

1. preprocess_pipeline.py for controlled noise context generation
2. eval_llm_gpt4.py for evaluation
3. infer_llm_ckplug.py for CK-PLUG method
4. infer_llm_vllm.py for batch generate
5. infer_llm_idiom_meaning_logits_attention_weight.py for generating with idiom entropy
6. infer_llm_idiom_meaning_logits_attention_A.py for generating with attention weight
7. attention_analysis.py for analyzing attention weight
