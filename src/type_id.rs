use std::collections::HashMap;
use crate::context::*;
use crate::params::*;
use crate::displayable_with_context::*;
use std::fmt;
use rand::prelude::*;

///A directory of [`Type`]s associating them to [`TypeId`]s, used as
///part of the specification of a [`Context`]. There can be multiple
///`Type::VecType`s with different [`TypeId`]s, but given argument
///and return [`TypeId`]s, this directory may only contain at most one
///[`TypeId`] with the corresponding `Type::FuncType`.
pub struct TypeInfoDirectory {
    info_vec : Vec::<Type>,
    func_ind_map : HashMap<(TypeId, TypeId), TypeId>,
    ret_map : HashMap::<TypeId, Vec::<(TypeId, TypeId)>>
}

impl TypeInfoDirectory {
    ///Creates an empty [`TypeInfoDirectory`].
    pub fn new() -> Self {
        TypeInfoDirectory {
            info_vec : Vec::new(),
            func_ind_map : HashMap::new(),
            ret_map : HashMap::new()
        }
    }
    ///Adds the given [`Type`] to this [`TypeInfoDirectory`], and
    ///returns the [`TypeId`] that it was assigned.
    pub fn add(&mut self, info : Type) -> TypeId {
        let added_type_id : usize = self.info_vec.len();

        self.ret_map.insert(added_type_id, Vec::new());

        if let Type::FuncType(arg_type, ret_type) = info {
            let pair = (arg_type, ret_type);
            if (self.func_ind_map.contains_key(&pair)) {
                return *self.func_ind_map.get(&pair).unwrap();
            } else {
                self.func_ind_map.insert(pair, added_type_id);
            }

            let ret_row = self.ret_map.get_mut(&ret_type).unwrap();
            ret_row.push((added_type_id, arg_type));
        }

        self.info_vec.push(info); 
        added_type_id
    }
    ///Gets the total number of types stored in this [`TypeInfoDirectory`].
    ///Since [`TypeId`]s are allocated consecutively, it's correct to use
    ///this as an iteration bound if you want to iterate over all [`TypeId`]s stored here.
    pub fn get_total_num_types(&self) -> usize {
        self.info_vec.len()
    }
    ///Returns true iff this [`TypeInfoDirectory`] has a registered [`TypeId`] for
    ///`Type::FuncType(arg_type_id, ret_type_id)`.
    pub fn has_func_type(&self, arg_type_id : TypeId, ret_type_id : TypeId) -> bool {
        let pair = (arg_type_id, ret_type_id);
        self.func_ind_map.contains_key(&pair)
    }
    ///Assuming that there is a [`TypeId`] for a function from the given argument [`TypeId`]
    ///to the given return [`TypeId`], yields the [`TypeId`] of the function type.
    pub fn get_func_type_id(&self, arg_type_id : TypeId, ret_type_id : TypeId) -> TypeId {
        let pair = (arg_type_id, ret_type_id);
        *self.func_ind_map.get(&pair).unwrap()
    }
    ///Gets the [`Type`] information stored for the given [`TypeId`].
    pub fn get_type(&self, id : TypeId) -> Type {
        self.info_vec[id]
    }
    ///Gets all `(func_type_id, arg_type_id)` pairs which may be applied to yield the
    ///given [`TypeId`].
    pub fn get_application_type_ids(&self, id : TypeId) -> Vec::<(TypeId, TypeId)> {
        self.ret_map.get(&id).unwrap().clone()
    }
    ///Given the [`TypeId`] of a function type, yields the [`TypeId`] of the return type.
    pub fn get_ret_type_id(&self, func_type_id : TypeId) -> TypeId {
        let func_type = self.get_type(func_type_id);
        if let Type::FuncType(_, ret_type_id) = func_type {
            ret_type_id
        } else {
            panic!();
        }
    }
    ///Returns true iff the given [`TypeId`] points to a `Type::VecType`.
    pub fn is_vector_type(&self, id : TypeId) -> bool {
        let kind = self.get_type(id);
        match (kind) {
            Type::FuncType(_, _) => false,
            Type::VecType(_) => true
        }
    }
    ///Given the [`TypeId`] of a function type, yields the [`TypeId`] of the argument type.
    pub fn get_arg_type_id(&self, func_type_id : TypeId) -> TypeId {
        let func_type = self.get_type(func_type_id);
        if let Type::FuncType(arg_type_id, _) = func_type {
            arg_type_id
        } else {
            panic!();
        }
    }
    ///Assuming that the given [`TypeId`] points to a `Type::VecType`, yields the
    ///declared number of dimensions for that type's base space.
    pub fn get_dimension(&self, vec_type_id : TypeId) -> usize {
        let vec_type = self.get_type(vec_type_id);
        if let Type::VecType(dim) = vec_type {
            dim
        } else {
            panic!();
        }
    }
}

///Type alias for a `usize` index into a [`TypeInfoDirectory`]'s storage of
///[`Type`]s. These are assumed to uniquely identify a type, whereas there may
///be multiple [`TypeId`]s for the same [`Type`]. This is done so that
///different vector types of the same dimension can potentially have different
///canonical feature mappings, for instance, if one represents 2d images, and
///the other represents 1d signals.
pub type TypeId = usize;

///Fundamental information about a type, generally indexed by a [`TypeId`] and
///stored in a [`TypeInfoDirectory`].
#[derive(Copy, Eq, PartialEq, Hash, Debug, Clone)]
pub enum Type {
    ///A type for vectors with the given declared number of dimensions for their base space
    VecType(usize),
    ///A type for functions which map elements of the former [`TypeId`] to the latter [`TypeId`],
    ///which is consequently only truly meaningful in the context of a [`TypeInfoDirectory`].
    FuncType(TypeId, TypeId)
}

impl DisplayableWithContext for Type {
    fn display(&self, ctxt : &Context) -> String {
        match (self) {
            Type::VecType(n) => format!("{}", n),
            Type::FuncType(arg, ret) => format!("({} -> {})", 
                                        ctxt.get_type(*arg).display(ctxt), 
                                        ctxt.get_type(*ret).display(ctxt))
        }
    }
}
