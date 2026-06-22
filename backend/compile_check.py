import compileall
import sys

print("Verifying Python syntax compile check on app directory...")
success = compileall.compile_dir("backend/app", force=True, quiet=False)

if success:
    print("\n[SUCCESS] All backend/app python files compiled successfully!")
    sys.exit(0)
else:
    print("\n[ERROR] Compilation errors detected in backend/app!")
    sys.exit(1)
