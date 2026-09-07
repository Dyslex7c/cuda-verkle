use ark_ed_on_bls12_381_bandersnatch::{Fq, Fr};
use ark_ff::{BigInteger, Field, PrimeField, UniformRand, Zero, One};
use banderwagon::{multi_scalar_mul, Element};
use ipa_multipoint::crs::CRS;
use rand::SeedableRng;
use rand::rngs::StdRng;
use serde::Serialize;
use std::env;
use std::fs;
use std::path::Path;

#[derive(Serialize)]
struct FieldTestVectors {
    field_name: String,
    modulus: String,
    test_cases: Vec<FieldTestCase>,
}

#[derive(Serialize)]
struct FieldTestCase {
    op: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    a: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    b: Option<String>,
    result: String,
}

#[derive(Serialize)]
struct CompressedPoint {
    bytes: String,
}

#[derive(Serialize)]
struct CurveTestVectors {
    curve: String,
    generator: CompressedPoint,
    test_cases: Vec<CurveTestCase>,
}

#[derive(Serialize)]
struct CurveTestCase {
    op: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    point: Option<CompressedPoint>,
    #[serde(skip_serializing_if = "Option::is_none")]
    scalar: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    a: Option<CompressedPoint>,
    #[serde(skip_serializing_if = "Option::is_none")]
    b: Option<CompressedPoint>,
    #[serde(skip_serializing_if = "Option::is_none")]
    result: Option<CompressedPoint>,
    #[serde(skip_serializing_if = "Option::is_none")]
    result_bytes: Option<String>,
}

#[derive(Serialize)]
struct CommitmentTestVectors {
    description: String,
    crs_description: String,
    test_cases: Vec<CommitmentTestCase>,
}

#[derive(Serialize)]
struct CommitmentTestCase {
    name: String,
    scalars: Vec<String>,
    commitment: String,
}

#[derive(Serialize)]
struct TreeTestVectors {
    description: String,
    tree_depth: usize,
    width: usize,
    test_cases: Vec<TreeTestCase>,
}

#[derive(Serialize)]
struct Update {
    index: usize,
    new_value: String,
}

#[derive(Serialize)]
struct TreeTestCase {
    name: String,
    initial_leaves: Vec<String>,
    updates: Vec<Update>,
    expected_root: String,
}

#[derive(Serialize)]
struct GroupToScalarTestVectors {
    description: String,
    test_cases: Vec<GroupToScalarTestCase>,
}

#[derive(Serialize)]
struct GroupToScalarTestCase {
    name: String,
    point: String,
    scalar: String,
}

#[derive(Serialize)]
struct TreeDepth2TestVectors {
    description: String,
    tree_depth: usize,
    width: usize,
    test_cases: Vec<TreeDepth2TestCase>,
}

#[derive(Serialize)]
struct TreeDepth2TestCase {
    name: String,
    updates: Vec<Update>,
    expected_root: String,
}

fn fq_to_hex(f: &Fq) -> String {
    let raw = f.into_bigint().to_bytes_be();
    assert!(raw.len() <= 32, "Fq does not fit in 32 bytes");
    let mut bytes = vec![0u8; 32 - raw.len()];
    bytes.extend(raw);
    hex::encode(bytes)
}

fn fr_to_hex(f: &Fr) -> String {
    let raw = f.into_bigint().to_bytes_be();
    assert!(raw.len() <= 32, "Fr does not fit in 32 bytes");
    let mut bytes = vec![0u8; 32 - raw.len()];
    bytes.extend(raw);
    hex::encode(bytes)
}

fn compressed_point(p: &Element) -> CompressedPoint {
    let bytes = p.to_bytes();
    CompressedPoint {
        bytes: hex::encode(bytes),
    }
}

fn generate_field_tests(rng: &mut StdRng) -> FieldTestVectors {
    let mut test_cases = Vec::new();
    
    let zero = Fq::zero();
    let one = Fq::one();
    let mut p_minus_one = Fq::zero();
    p_minus_one -= &one;

    let values = vec![
        zero, one, p_minus_one,
        Fq::from(2u64), Fq::from(100u64),
    ];

    for &a in &values {
        for &b in &values {
            test_cases.push(FieldTestCase {
                op: "add".into(),
                a: Some(fq_to_hex(&a)),
                b: Some(fq_to_hex(&b)),
                result: fq_to_hex(&(a + b)),
            });
            test_cases.push(FieldTestCase {
                op: "mul".into(),
                a: Some(fq_to_hex(&a)),
                b: Some(fq_to_hex(&b)),
                result: fq_to_hex(&(a * b)),
            });
            test_cases.push(FieldTestCase {
                op: "sub".into(),
                a: Some(fq_to_hex(&a)),
                b: Some(fq_to_hex(&b)),
                result: fq_to_hex(&(a - b)),
            });
        }
        
        test_cases.push(FieldTestCase {
            op: "sqr".into(),
            a: Some(fq_to_hex(&a)),
            b: None,
            result: fq_to_hex(&(a * a)),
        });
        
        if let Some(inv) = a.inverse() {
            test_cases.push(FieldTestCase {
                op: "inv".into(),
                a: Some(fq_to_hex(&a)),
                b: None,
                result: fq_to_hex(&inv),
            });
        }
    }

    // Add some random tests
    for _ in 0..10 {
        let a = Fq::rand(rng);
        let b = Fq::rand(rng);
        test_cases.push(FieldTestCase {
            op: "add".into(),
            a: Some(fq_to_hex(&a)),
            b: Some(fq_to_hex(&b)),
            result: fq_to_hex(&(a + b)),
        });
        test_cases.push(FieldTestCase {
            op: "mul".into(),
            a: Some(fq_to_hex(&a)),
            b: Some(fq_to_hex(&b)),
            result: fq_to_hex(&(a * b)),
        });
        test_cases.push(FieldTestCase {
            op: "inv".into(),
            a: Some(fq_to_hex(&a)),
            b: None,
            result: fq_to_hex(&(a.inverse().unwrap_or(zero))),
        });
    }

    FieldTestVectors {
        field_name: "BLS12-381 scalar field (Fq for Bandersnatch)".into(),
        modulus: "73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001".into(),
        test_cases,
    }
}

fn generate_curve_tests(_rng: &mut StdRng) -> CurveTestVectors {
    let generator = Element::prime_subgroup_generator();
    let mut test_cases = Vec::new();

    // gen * 1
    test_cases.push(CurveTestCase {
        op: "scalar_mul".into(),
        point: Some(compressed_point(&generator)),
        scalar: Some(fr_to_hex(&Fr::one())),
        a: None,
        b: None,
        result: Some(compressed_point(&generator)),
        result_bytes: None,
    });

    let gen_times_2 = generator + generator;
    test_cases.push(CurveTestCase {
        op: "add".into(),
        point: None,
        scalar: None,
        a: Some(compressed_point(&generator)),
        b: Some(compressed_point(&generator)),
        result: Some(compressed_point(&gen_times_2)),
        result_bytes: None,
    });

    // to_bytes
    test_cases.push(CurveTestCase {
        op: "to_bytes".into(),
        point: Some(compressed_point(&generator)),
        scalar: None,
        a: None,
        b: None,
        result: None,
        result_bytes: Some(hex::encode(generator.to_bytes())),
    });

    CurveTestVectors {
        curve: "Bandersnatch/Banderwagon".into(),
        generator: compressed_point(&generator),
        test_cases,
    }
}

fn generate_commitment_tests(rng: &mut StdRng) -> CommitmentTestVectors {
    let crs = CRS::default();
    let mut test_cases = Vec::new();

    let all_zeros = vec![Fr::zero(); 256];
    let comm_zeros = multi_scalar_mul(&crs.G[..256], &all_zeros);
    test_cases.push(CommitmentTestCase {
        name: "all_zeros".into(),
        scalars: all_zeros.iter().map(fr_to_hex).collect(),
        commitment: hex::encode(comm_zeros.to_bytes()),
    });

    let mut single_one_0 = vec![Fr::zero(); 256];
    single_one_0[0] = Fr::one();
    let comm_one_0 = multi_scalar_mul(&crs.G[..256], &single_one_0);
    test_cases.push(CommitmentTestCase {
        name: "single_one_at_0".into(),
        scalars: single_one_0.iter().map(fr_to_hex).collect(),
        commitment: hex::encode(comm_one_0.to_bytes()),
    });

    // Full-width deterministic scalars exercise all scalar windows and give
    // the C++ suite a differential check against rust-verkle's independent
    // MSM implementation, rather than only boundary-value fixtures.
    for case_index in 0..4 {
        let scalars: Vec<Fr> = (0..256).map(|_| Fr::rand(rng)).collect();
        let commitment = multi_scalar_mul(&crs.G[..256], &scalars);
        test_cases.push(CommitmentTestCase {
            name: format!("random_full_width_{case_index}"),
            scalars: scalars.iter().map(fr_to_hex).collect(),
            commitment: hex::encode(commitment.to_bytes()),
        });
    }

    CommitmentTestVectors {
        description: "256-wide Pedersen commitment test vectors".into(),
        crs_description: "From rust-verkle default CRS".into(),
        test_cases,
    }
}

fn generate_tree_tests(rng: &mut StdRng) -> TreeTestVectors {
    let crs = CRS::default();
    let mut test_cases = Vec::new();

    let mut initial_leaves = vec![Fr::zero(); 256];
    for i in 0..256 {
        initial_leaves[i] = Fr::rand(rng);
    }

    let mut leaves = initial_leaves.clone();
    leaves[0] = Fr::rand(rng);
    
    let comm = multi_scalar_mul(&crs.G[..256], &leaves);

    test_cases.push(TreeTestCase {
        name: "single_update".into(),
        initial_leaves: initial_leaves.iter().map(fr_to_hex).collect(),
        updates: vec![Update {
            index: 0,
            new_value: fr_to_hex(&leaves[0]),
        }],
        expected_root: hex::encode(comm.to_bytes()),
    });

    TreeTestVectors {
        description: "Verkle tree incremental update test vectors".into(),
        tree_depth: 1,
        width: 256,
        test_cases,
    }
}

fn generate_group_to_scalar_tests(rng: &mut StdRng) -> GroupToScalarTestVectors {
    let crs = CRS::default();
    let zero_scalars = vec![Fr::zero(); 256];
    let identity = multi_scalar_mul(&crs.G[..256], &zero_scalars);
    let generator = crs.G[0];
    let sum = generator + crs.G[1];
    let random_scalars: Vec<Fr> = (0..256).map(|_| Fr::rand(rng)).collect();
    let random_commitment = multi_scalar_mul(&crs.G[..256], &random_scalars);
    let points = [
        ("identity", identity),
        ("crs_g0", generator),
        ("crs_g0_plus_g1", sum),
        ("random_commitment", random_commitment),
    ];

    GroupToScalarTestVectors {
        description: "EIP-6800 group_to_scalar_field vectors from rust-verkle map_to_scalar_field".into(),
        test_cases: points.into_iter().map(|(name, point)| GroupToScalarTestCase {
            name: name.into(),
            point: hex::encode(point.to_bytes()),
            scalar: fr_to_hex(&point.map_to_scalar_field()),
        }).collect(),
    }
}

fn generate_depth2_tree_tests(rng: &mut StdRng) -> TreeDepth2TestVectors {
    let crs = CRS::default();
    let mut leaves = vec![Fr::zero(); 256 * 256];
    let indices = [0usize, 255, 256, 257, 17 * 256 + 99, 255 * 256 + 255];
    let updates: Vec<Update> = indices.into_iter().map(|index| {
        let new_value = Fr::rand(rng);
        leaves[index] = new_value;
        Update { index, new_value: fr_to_hex(&new_value) }
    }).collect();
    let l1_commitments: Vec<Element> = leaves.chunks(256)
        .map(|children| multi_scalar_mul(&crs.G[..256], children))
        .collect();
    let l1_scalars: Vec<Fr> = l1_commitments.iter()
        .map(Element::map_to_scalar_field)
        .collect();
    let root = multi_scalar_mul(&crs.G[..256], &l1_scalars);

    TreeDepth2TestVectors {
        description: "Depth-2 EIP-6800 group_to_scalar_field tree vector from rust-verkle".into(),
        tree_depth: 2,
        width: 256,
        test_cases: vec![TreeDepth2TestCase {
            name: "sparse_cross_branch_updates".into(),
            updates,
            expected_root: hex::encode(root.to_bytes()),
        }],
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let subcommand = if args.len() > 1 { &args[1] } else { "generate" };

    if subcommand == "generate" {
        eprintln!("Generating test vectors...");
        
        let out_dir = Path::new("../test_vectors");
        if !out_dir.exists() {
            fs::create_dir_all(out_dir).unwrap();
        }

        let mut rng = StdRng::seed_from_u64(42);

        // field Tests
        let field_tests = generate_field_tests(&mut rng);
        let field_json = serde_json::to_string_pretty(&field_tests).unwrap();
        fs::write(out_dir.join("field_test_vectors.json"), field_json).unwrap();
        eprintln!("Generated field_test_vectors.json");

        // curve Tests
        let curve_tests = generate_curve_tests(&mut rng);
        let curve_json = serde_json::to_string_pretty(&curve_tests).unwrap();
        fs::write(out_dir.join("curve_test_vectors.json"), curve_json).unwrap();
        eprintln!("Generated curve_test_vectors.json");

        // commitment Tests
        let comm_tests = generate_commitment_tests(&mut rng);
        let comm_json = serde_json::to_string_pretty(&comm_tests).unwrap();
        fs::write(out_dir.join("commitment_test_vectors.json"), comm_json).unwrap();
        eprintln!("Generated commitment_test_vectors.json");

        // tree Tests
        let tree_tests = generate_tree_tests(&mut rng);
        let tree_json = serde_json::to_string_pretty(&tree_tests).unwrap();
        fs::write(out_dir.join("tree_test_vectors.json"), tree_json).unwrap();
        eprintln!("Generated tree_test_vectors.json");

        let group_to_scalar_tests = generate_group_to_scalar_tests(&mut rng);
        let group_to_scalar_json = serde_json::to_string_pretty(&group_to_scalar_tests).unwrap();
        fs::write(out_dir.join("group_to_scalar_test_vectors.json"), group_to_scalar_json).unwrap();
        eprintln!("Generated group_to_scalar_test_vectors.json");

        let depth2_tree_tests = generate_depth2_tree_tests(&mut rng);
        let depth2_tree_json = serde_json::to_string_pretty(&depth2_tree_tests).unwrap();
        fs::write(out_dir.join("tree_depth2_test_vectors.json"), depth2_tree_json).unwrap();
        eprintln!("Generated tree_depth2_test_vectors.json");

        eprintln!("All test vectors generated successfully in ../test_vectors/");
    } else {
        eprintln!("Unknown subcommand: {}", subcommand);
        eprintln!("Usage: {} [generate]", args[0]);
    }
}
