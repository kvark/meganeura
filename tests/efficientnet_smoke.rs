//! Smoke test for the EfficientNetV2-S features[0:6] builder.
//!
//! Phase 2a/2b coverage: confirms the graph constructs, all parameter
//! names match a known-good list (so the loader in Phase 2c can map
//! 1:1 from `efficientnet_v2_s` checkpoints), and the output shape is
//! the expected `[batch * 160 * 12 * 12]`.
//!
//! Numerical parity against torchvision lives in Phase 2c (alongside
//! the BN-fold weight loader); this test only checks the structural
//! contract.

use meganeura::Graph;
use meganeura::models::efficientnet;

#[test]
fn efficientnet_v2s_features_graph_constructs() {
    let mut g = Graph::new();
    let out = efficientnet::build_graph(&mut g, /*batch=*/ 1);

    let out_node = g.node(out);
    assert_eq!(
        out_node.ty.shape,
        vec![160 * 12 * 12],
        "EfficientNetV2-S features[0:6] on 192×192 must output [N=1, C=160, H=12, W=12]"
    );
}

#[test]
fn efficientnet_v2s_weight_names_cover_torchvision_keys() {
    let names = efficientnet::weight_names();

    // Sentinel keys spanning the three block kinds: stem, FusedMBConv
    // (e=1 and e=4), and MBConv-with-SE.  Verbatim from torchvision
    // `efficientnet_v2_s().state_dict()`.
    let expected_present = [
        "features.0.0.weight",
        "features.1.0.block.0.0.weight", // FusedMBConv e=1: single 3×3 (24, 24, 3, 3)
        "features.1.1.block.0.0.weight",
        "features.2.0.block.0.0.weight", // FusedMBConv e=4 expand (96, 24, 3, 3)
        "features.2.0.block.1.0.weight", // FusedMBConv e=4 project (48, 96, 1, 1)
        "features.3.0.block.0.0.weight", // FusedMBConv e=4 (192, 48, 3, 3)
        "features.4.0.block.0.0.weight", // MBConv expand 1×1 (256, 64, 1, 1)
        "features.4.0.block.1.0.weight", // MBConv depthwise (256, 1, 3, 3)
        "features.4.0.block.2.fc1.weight", // MBConv SE squeeze 256→16
        "features.4.0.block.3.0.weight", // MBConv project (128, 256, 1, 1)
        "features.5.8.block.3.0.weight", // last block project (160, 960, 1, 1)
    ];
    for needle in expected_present {
        assert!(
            names.iter().any(|n| n == needle),
            "weight_names() missing '{needle}'"
        );
    }

    // FusedMBConv stages (1, 2, 3) must NOT advertise SE branch keys.
    let no_se_keys = [
        "features.1.0.block.2.fc1.weight",
        "features.2.0.block.2.fc1.weight",
        "features.3.0.block.2.fc1.weight",
    ];
    for forbidden in no_se_keys {
        assert!(
            !names.iter().any(|n| n == forbidden),
            "weight_names() should not include SE key '{forbidden}' in FusedMBConv stages"
        );
    }
}
