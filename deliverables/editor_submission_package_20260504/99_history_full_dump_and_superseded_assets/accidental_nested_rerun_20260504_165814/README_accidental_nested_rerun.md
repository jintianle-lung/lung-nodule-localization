# Accidental Nested Rerun Note

During cleanup, `finalize_editor_submission_package.py` was briefly run from its package-local copy before the script had repo-root discovery. It created a small nested `deliverables/editor_submission_package_20260504` output. Nothing was deleted; the nested output was moved here for traceability.

The script has since been fixed to locate the repository root by walking upward to `.git`.

The tiny nested ZIP created during that accidental run was moved out of the package to `deliverables/_local_zip_archives_not_for_git_20260504/accidental_nested_rerun_20260504_165814_editor_submission_package_20260504.zip`, so no ZIP files are kept inside the git-tracked submission package.
