---
inclusion: auto
---

# Agent Working Guidelines

## Documentation Philosophy
- **Centralize**: All agent docs in `.kiro/agent/`, NOT project root
- **Minimize**: Few well-maintained files > many scattered files
- **Update**: Keep existing docs current instead of creating new ones
- **Brevity**: Short, actionable content only

## Code Modifications
- **Test first**: Use `configs/quick_test.yaml` for fast verification
- **Preserve**: Maintain backward compatibility
- **Modular**: Keep src/ files focused and single-purpose
- **No breaking changes**: Unless explicitly requested

## File Management
- **User docs**: Only in `docs/` (for end users)
- **Agent context**: Only in `.kiro/agent/` (for AI assistant)
- **Root files**: Minimize - only essential project files
- **Update vs Create**: Always prefer updating existing files

## Communication Style
- **Concise**: Brief explanations, no verbose summaries
- **Actionable**: Focus on what to do, not what was done
- **Technical**: Assume developer audience
- **No repetition**: Don't restate what's already clear

## When Making Changes
1. Check `.kiro/agent/status.md` for current context
2. Update relevant files in `.kiro/agent/` as you work
3. Keep changes focused and minimal
4. Test with quick_test.yaml if modifying core code
5. Update status.md with completion notes

## What NOT to Do
- ❌ Create markdown summaries in project root
- ❌ Generate verbose documentation after simple tasks
- ❌ Create new files when existing ones can be updated
- ❌ Repeat information already in project docs
- ❌ Make breaking changes without explicit approval

## What TO Do
- ✅ Keep `.kiro/agent/status.md` current
- ✅ Update existing docs when relevant
- ✅ Make minimal, focused changes
- ✅ Test changes before declaring complete
- ✅ Maintain backward compatibility
