# TODOs

## Config Refactoring

- **DatasetConfig discriminated union**: Refactor `DatasetConfig` to use discriminated unions like `BackendConfig`. Currently, type-specific fields (e.g., `conversation_turns` for ShareGPT, `output_len` for random, `prompts` for inline) are flat with fields silently ignored when not applicable. A proper discriminated union would provide type safety and clearer documentation of which fields apply to which dataset types.
