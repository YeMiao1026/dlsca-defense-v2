import numpy as np

from src.leakage_probe import AES_SBOX, compute_leakage_labels


def test_compute_leakage_labels_matches_manual_sbox_and_mask():
    meta = np.zeros(4, dtype=[("plaintext", "u1", (16,)), ("key", "u1", (16,)), ("masks", "u1", (16,))])
    meta["plaintext"][:, 2] = [0x00, 0x01, 0xFF, 0x42]
    meta["key"][:, 2] = [0x00, 0x00, 0x00, 0x11]
    meta["masks"][:, 0] = [10, 20, 30, 40]

    labels = compute_leakage_labels(meta, target_byte=2, mask_index=0)

    expected_unmasked = AES_SBOX[meta["plaintext"][:, 2] ^ meta["key"][:, 2]]
    expected_masked = (expected_unmasked ^ meta["masks"][:, 0]).astype(np.int32)
    np.testing.assert_array_equal(labels["masked_value"], expected_masked)
    np.testing.assert_array_equal(labels["mask_value"], [10, 20, 30, 40])
