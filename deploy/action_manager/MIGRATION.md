# Action Manager Refactoring - Migration Guide

## Summary of Changes

The action manager module has been refactored from a single file (`deploy/action_manager.py`) into a modular package structure under `deploy/action_manager/` with configuration-based initialization.

## What Changed

### 1. File Structure

**Before:**
```
deploy/
  action_manager.py  # All managers in one file
```

**After:**
```
deploy/
  action_manager/
    __init__.py           # Package exports
    base.py              # AbstractActionManager
    basic.py             # BasicActionManager
    older_first.py       # OlderFirstManager
    temporal_agg.py      # TemporalAggManager
    temporal_older.py    # TemporalOlderManager
    delay_free.py        # DelayFreeManager
    loader.py            # load_action_manager() function
    MIGRATION.md         # This file

configs/
  action_manager/
    basic.yaml           # BasicActionManager config
    older_first.yaml     # OlderFirstManager config
    temporal_agg.yaml    # TemporalAggManager config
    temporal_older.yaml  # TemporalOlderManager config
    delay_free.yaml      # DelayFreeManager config
    README.md            # Configuration documentation
```

### 2. Import Changes

**Import remains the same:**
```python
from deploy.action_manager import load_action_manager
```

The package structure is transparent to existing code!

### 3. Initialization Changes

**Before (old way - still works for backward compatibility):**
```python
# Using class name string
action_manager = load_action_manager('OlderFirstManager', args)
```

**After (new recommended way):**
```python
# Using config name
action_manager = load_action_manager('older_first', args)

# Using config file
action_manager = load_action_manager('configs/action_manager/older_first.yaml', args)
```

### 4. Parameter Handling

**Before:**
```bash
python eval_real.py \
    --action_manager OlderFirstManager \
    --manager_coef 1.0
```

**After (more flexible):**
```bash
# Using config name
python eval_real.py \
    --action_manager older_first \
    --manager_coef 1.0

# Using config file (parameters in YAML)
python eval_real.py \
    --action_manager configs/action_manager/older_first.yaml

# Override parameters from config file
python eval_real.py \
    --action_manager older_first \
    --manager_coef 0.8 \
    --older_coef 0.7
```

## Benefits

1. **Modularity**: Each manager is in its own file, easier to understand and maintain
2. **Configuration**: Managers can be configured via YAML files
3. **Flexibility**: Support for both config names and file paths
4. **Documentation**: Each config file documents its parameters
5. **Extensibility**: Easy to add new managers without modifying existing code
6. **Backward Compatible**: Old class-name based loading still works

## Migration Steps

### For End Users

**No changes required!** Your existing scripts will continue to work.

If you want to use the new config-based approach:

1. Replace class names with config names:
   - `OlderFirstManager` → `older_first`
   - `BasicActionManager` → `basic`
   - `TemporalAggManager` → `temporal_agg`
   - `TemporalOlderManager` → `temporal_older`
   - `DelayFreeManager` → `delay_free`

2. Use additional parameter flags as needed:
   - `--manager_coef` for coefficient override
   - `--older_coef` for older coefficient
   - `--duration` for delay-free duration

### For Developers Adding New Managers

See the "Customization" section in `docs/12_action_manager.md` for detailed instructions.

Quick overview:
1. Create new manager class in `deploy/action_manager/my_manager.py`
2. Register in `deploy/action_manager/__init__.py`
3. Add to `MANAGER_MAP` in `deploy/action_manager/loader.py`
4. Create config file in `configs/action_manager/my_manager.yaml` (optional)

## Bug Fixes

- **Removed `exit()` call** in `OlderFirstManager.__init__()` that was preventing normal execution
- **Fixed parameter naming**: All managers now consistently use `coef` instead of `manager_coef` internally

## Examples

### Example 1: Basic usage
```bash
python eval_real.py --action_manager basic
```

### Example 2: Older-first with 75% threshold
```bash
python eval_real.py --action_manager older_first --manager_coef 0.75
```

### Example 3: Temporal aggregation with light smoothing
```bash
python eval_real.py --action_manager temporal_agg --manager_coef 0.1
```

### Example 4: Custom config file
```bash
# Create custom config
cat > configs/action_manager/my_tuned.yaml << EOF
name: my_tuned
manager_name: TemporalOlderManager
coef: 0.12
older_coef: 0.8
EOF

# Use it
python eval_real.py --action_manager my_tuned
```

## Troubleshooting

### Q: My script fails with "Unknown action manager"
**A:** Check your spelling. Use `older_first` not `OlderFirst`. Or use the full class name `OlderFirstManager` for backward compatibility.

### Q: Parameters from config file are not being used
**A:** Command-line parameters override config file parameters. Check your command-line arguments.

### Q: Import error when loading action_manager
**A:** This is likely due to missing dependencies in the project (e.g., `tianshou`). This is not related to the refactoring. The action_manager module itself has no external dependencies.

## See Also

- `configs/action_manager/README.md` - Configuration guide and parameter reference
- `docs/12_action_manager.md` - Conceptual overview and usage guide

