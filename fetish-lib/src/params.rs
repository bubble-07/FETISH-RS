///Maximum-singular-value-relative truncation threshold for all pseudoinverses
pub const PINV_TRUNCATION_THRESH : f32 = 0.0001f32;

///Truncation threshold for squared norms of data-point inputs to a regression,
///below which we assume that no regression information is actually being given.
pub const UPDATE_SQ_NORM_TRUNCATION_THRESH : f32 = 0.0000001f32;

///Default threshold for floating-point-equality checks in [`crate::test_utils`].
pub const DEFAULT_TEST_THRESH : f32 = 0.001f32;
