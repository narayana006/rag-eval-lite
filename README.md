# rag-eval-lite

Measure RAG quality on your own documents: synthetic QA generation, retrieval recall and answer faithfulness, HTML report.

> **Project status: planning. There is no working code in this repository yet.**
> This README describes what the project will do. Nothing below can be installed or run today.
> Features are ticked off in the checklist as they are built.

## What it is

A Python library and command-line tool.

**The problem it solves:** Most RAG systems ship without any measurement of whether retrieval or answers are actually good.

## Who it is for

Teams building retrieval-augmented generation (RAG) who need numbers instead of impressions.

## How it will work

1. The tool reads your documents and generates questions with known answers and known source passages.
2. It runs those questions through your retriever and your answer step.
3. It scores whether the right passages were retrieved and whether answers stick to the sources, then writes an HTML report you can compare between runs.

## Planned features

Unticked items are not built yet.

- [ ] Generate a question/answer set from your corpus
- [ ] Retrieval metrics: recall@k, MRR
- [ ] Answer faithfulness and relevance scoring
- [ ] Pluggable retriever interface (any vector DB or search engine)
- [ ] Single-file HTML report to compare runs

## Planned usage

This is the intended interface, shown so you can judge the design. **It does not work yet**, and names may change before the first release.

```bash
rag-eval-lite generate ./docs --n 100
rag-eval-lite score --retriever my_retriever:search
```

## Contributing

Feedback on the design is the most useful contribution right now. Open an issue to suggest a feature, question an approach or describe your use case. Code contributions are welcome once the first version lands.

## Author

Built and maintained by [Narayana Vemburaj](https://github.com/narayana006).

## License

[MIT](LICENSE)
