---
inclusion: auto
fileMatchPattern: "**/*.py"
---

# Python Code Standards

## Style
- Follow existing code style in the file
- Use type hints for function signatures
- Keep functions focused and single-purpose
- Docstrings for public functions only

## Structure
- Imports: stdlib → third-party → local
- Keep files under 500 lines
- Extract reusable logic to utils/
- No circular imports

## Data Handling
- Use pandas DataFrames for metadata
- Preserve column names: path, label, source, quality, food_category, defect_type, generator
- Always handle None/NaN in metadata gracefully

## PyTorch Conventions
- Use `@torch.no_grad()` for inference
- `.to(device, non_blocking=True)` for tensors
- `model.eval()` before evaluation
- Clear gradients with `zero_grad(set_to_none=True)`

## Error Handling
- Fail fast with clear error messages
- Use assertions for invariants
- Provide actionable error messages
- Log warnings for non-critical issues

## Testing
- Use `configs/quick_test.yaml` for fast iteration
- Test with small batch_size first
- Verify output files are generated
- Check CSV columns match expected schema

## Performance
- Use DataLoader with pin_memory=True on GPU
- Batch operations when possible
- Avoid unnecessary file I/O
- Cache expensive computations

## Compatibility
- Maintain backward compatibility with old dataset
- Keep legacy functions available
- Add new features without breaking existing code
