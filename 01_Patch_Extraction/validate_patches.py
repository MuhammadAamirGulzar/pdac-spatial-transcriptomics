#!/usr/bin/env python
"""
Wrapper script to run validation pipeline as a module
"""

import sys
from pathlib import Path

# Add package to path
package_dir = Path(__file__).parent / "patch_st_validation"
sys.path.insert(0, str(package_dir))

# Now run main
if __name__ == "__main__":
    from patch_st_validation.main import run_validation_pipeline
    try:
        run_validation_pipeline()
    except KeyboardInterrupt:
        print("\nValidation pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
