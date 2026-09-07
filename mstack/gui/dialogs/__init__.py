"""Standalone dialogs shared by the collector GUIs."""

from mstack.gui.dialogs.convert import LerobotConvertDialog
from mstack.gui.dialogs.hf_account import (
    HfAccountDialog,
    hf_account,
    hf_add_account,
    hf_stored_accounts,
    hf_switch_account,
)
from mstack.gui.dialogs.repack import RepackDialog
from mstack.gui.dialogs.schema import DatasetSchemaDialog
from mstack.gui.dialogs.upload import HdfUploadDialog

__all__ = [
    "DatasetSchemaDialog",
    "HfAccountDialog",
    "HdfUploadDialog",
    "LerobotConvertDialog",
    "RepackDialog",
    "hf_account",
    "hf_add_account",
    "hf_stored_accounts",
    "hf_switch_account",
]
