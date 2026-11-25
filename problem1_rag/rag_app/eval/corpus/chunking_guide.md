# Chunking Strategies for Retrieval-Augmented Generation

Document chunking splits long source files into retrieval units that fit within embedding model context windows and LLM prompt budgets. Two primary parameters control fixed-window chunking: `chunk_size` and `chunk_overlap`.

## Choosing Chunk Size and Overlap
A default chunk size of 500 characters with a 100-character overlap preserves sentence continuity across boundaries while keeping individual passages focused on a single concept. Smaller chunks such as 200 characters fragment multi-sentence explanations and reduce recall on complex questions. Larger chunks such as 1,200 characters dilute embedding specificity and waste LLM context tokens.

## Boundary-Aware Splitting
Rather than cutting mid-word, boundary-aware splitters scan backward within the final 20 percent of the target window for paragraph breaks, sentence periods, or whitespace. This simple heuristic improves answer faithfulness because retrieved passages remain grammatically complete.
