# Grounded Generation and Hallucination Prevention

Retrieval-Augmented Generation reduces factual hallucinations by conditioning the language model on retrieved source passages. However, if retrieval returns low-similarity or off-topic chunks, an unconstrained LLM will still fabricate plausible-sounding answers from parametric memory.

## Similarity Thresholding and Refusal
A production RAG pipeline must enforce a minimum cosine similarity threshold before invoking the generator. When zero retrieved chunks meet `min_score`, the service short-circuits to an explicit refusal message stating that no relevant context was found. This saves generation tokens and eliminates unsupported guesses on out-of-domain queries.

## Inline Chunk Citations
Every factual claim in the generated response should include an inline bracketed citation such as `[chunk_id]`. Requiring explicit citations allows downstream evaluators and users to verify faithfulness directly against the retrieved text.
