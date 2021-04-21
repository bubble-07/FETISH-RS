use crate::type_id::*;
use crate::space_info::*;
use crate::func_impl::*;
use crate::function_space_info::*;
use crate::feature_space_info::*;
use crate::primitive_directory::*;
use crate::primitive_term_pointer::*;

///Stores interpreter-global context information, such as the
///collection of all types in the language, the collection of all
///primitives, and the definitions of their associated featurization
///maps. See [`TypeInfoDirectory`], [`SpaceInfoDirectory`], [`PrimitiveDirectory`]
///for these individual components.
pub struct Context {
    pub type_info_directory : TypeInfoDirectory,
    pub space_info_directory : SpaceInfoDirectory,
    pub primitive_directory : PrimitiveDirectory
}

impl Context {
    //Primitive information
    
    ///Given a [`PrimitiveTermPointer`], yields the [`FuncImpl`] it references.
    pub fn get_primitive(&self, primitive_term_pointer : PrimitiveTermPointer) -> &dyn FuncImpl {
        self.primitive_directory.get_primitive(primitive_term_pointer)
    }

    //Space information
    
    ///Gets a reference to the [`FeatureSpaceInfo`] for the given [`TypeId`].
    pub fn get_feature_space_info(&self, type_id : TypeId) -> &FeatureSpaceInfo {
        self.space_info_directory.get_feature_space_info(type_id)
    }

    ///Gets the [`FunctionSpaceInfo`] for functions going from the given `arg_type_id` to the
    ///given `ret_type_id`.
    pub fn build_function_space_info(&self, arg_type_id : TypeId, ret_type_id : TypeId) -> FunctionSpaceInfo {
        let arg_feat_info = self.get_feature_space_info(arg_type_id);
        let ret_feat_info = self.get_feature_space_info(ret_type_id);
        FunctionSpaceInfo {
            in_feat_info : arg_feat_info,
            out_feat_info : ret_feat_info
        }
    }
    ///Gets the [`FunctionSpaceInfo`] for the given function [`TypeId`].
    pub fn get_function_space_info(&self, func_type_id : TypeId) -> FunctionSpaceInfo {
        let func_type = self.get_type(func_type_id);
        match (func_type) {
            Type::FuncType(arg_type_id, ret_type_id) => {
                let arg_feat_info = self.get_feature_space_info(arg_type_id);
                let ret_feat_info = self.get_feature_space_info(ret_type_id);
                FunctionSpaceInfo {
                    in_feat_info : arg_feat_info,
                    out_feat_info : ret_feat_info
                }
            },
            Type::VecType(_) => {
                panic!();
            }
        }
    }

    //Type Information
    ///Given the argument and result types for a function type, returns the
    ///[`TypeId`] of the corresponding function type, assuming that it exists.
    pub fn get_func_type_id(&self, arg_type_id : TypeId, ret_type_id : TypeId) -> TypeId {
        self.type_info_directory.get_func_type_id(arg_type_id, ret_type_id)
    }
    ///Given a [`TypeId`], yields the [`Type`] struct describing the type.
    pub fn get_type(&self, id : TypeId) -> Type {
        self.type_info_directory.get_type(id)
    }
    ///Given the argument and result types for a function type, returns true iff
    ///the function type actually exists in the [`TypeInfoDirectory`].
    pub fn has_func_type(&self, arg_type_id : TypeId, ret_type_id : TypeId) -> bool {
        self.type_info_directory.has_func_type(arg_type_id, ret_type_id)
    }
    ///Given a target type, yields the collection of all pairs `(func_type_id, arg_type_id)`
    ///in the [`TypeInfoDirectory`] for which the return type is the given target.
    pub fn get_application_type_ids(&self, id : TypeId) -> Vec::<(TypeId, TypeId)> {
        self.type_info_directory.get_application_type_ids(id)
    }
    ///Returns the total number of types registered in the [`TypeInfoDirectory`].
    pub fn get_total_num_types(&self) -> usize {
        self.type_info_directory.get_total_num_types()
    }
    ///Given the [`TypeId`] of a vector type, yields the dimensionality of the
    ///corresponding vector space.
    pub fn get_dimension(&self, vec_type_id : TypeId) -> usize {
        self.type_info_directory.get_dimension(vec_type_id)
    }
    ///Given the [`TypeId`] of a function type, yields the [`TypeId`] of the
    ///function's argument type.
    pub fn get_arg_type_id(&self, func_type_id : TypeId) -> TypeId {
        self.type_info_directory.get_arg_type_id(func_type_id)
    }
    ///Given the [`TypeId`] of a function type, yields the [`TypeId`] of the
    ///function's return type.
    pub fn get_ret_type_id(&self, func_type_id : TypeId) -> TypeId {
        self.type_info_directory.get_ret_type_id(func_type_id)
    }
    ///Given a [`TypeId`], returns `true` if the underlying [`Type`] is
    ///a vector type, and `false` if it's a function type instead.
    pub fn is_vector_type(&self, id : TypeId) -> bool {
        self.type_info_directory.is_vector_type(id)
    }
}
