# rag-eval-lite

> Measure RAG quality on your own documents: synthetic QA generation, retrieval recall and answer faithfulness, HTML report.

**Status:** 🚧 Early stage — design and roadmap published, implementation in progress. Stars, issues and ideas welcome.

## Why

Most RAG systems ship without any measurement of whether retrieval or answers are actually good.

## Planned features

- [ ] Generate a question/answer set from your corpus
- [ ] Retrieval metrics: recall@k, MRR
- [ ] Answer faithfulness and relevance scoring
- [ ] Pluggable retriever interface (any vector DB or search engine)
- [ ] Single-file HTML report to compare runs

## Intended usage

```
rag-eval-lite generate ./docs --n 100
rag-eval-lite score --retriever my_retriever:search
```

_The interface above is the target design and may change._

## Roadmap

1. Core implementation of the first feature set
2. Examples and tests
3. CI and first tagged release
4. Docs and integrations

## Contributing

Contributions are welcome. Open an issue to discuss an idea or pick an unchecked item from the feature list.

## License

[MIT](LICENSE)
