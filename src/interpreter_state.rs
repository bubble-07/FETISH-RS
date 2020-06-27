use std::collections::HashMap;
use crate::type_id::*;
use crate::application_table::*;
use crate::type_space::*;
use crate::term::*;
use crate::term_pointer::*;
use crate::term_reference::*;
use crate::term_application::*;
use crate::term_application_result::*;
use std::collections::HashSet;
use crate::func_impl::*;

pub struct InterpreterState {
    pub application_tables : HashMap::<TypeId, ApplicationTable>,
    pub type_spaces : HashMap::<TypeId, TypeSpace>
}

impl InterpreterState {

    pub fn store_term(mut self, type_id : TypeId, term : PartiallyAppliedTerm) -> (InterpreterState, TermPointer) {
        let type_space : &mut TypeSpace = self.type_spaces.get_mut(&type_id).unwrap();
        let result = type_space.add(term);
        (self, result)
    }

    pub fn get(&self, term_ptr : &TermPointer) -> &PartiallyAppliedTerm {
        self.type_spaces.get(&term_ptr.type_id).unwrap().get(term_ptr.index)
    }

    pub fn get_app_results_with_arg(&self, arg : &TermReference) -> Vec<TermApplicationResult> {
        let mut result : Vec<TermApplicationResult> = Vec::new();
        for table in self.application_tables.values() {
            let mut temp = table.get_app_results_with_arg(arg).clone();
            result.append(&mut temp);
        }
        result
    }

    pub fn get_app_results_with_func(&self, func : &TermPointer) -> Vec<TermApplicationResult> {
        let mut result : Vec<TermApplicationResult> = Vec::new();
        for table in self.application_tables.values() {
            let mut temp = table.get_app_results_with_func(func).clone();
            result.append(&mut temp);
        }
        result
    }

    pub fn get_app_results_with_result(&self, result_term : &TermReference) -> Vec<TermApplicationResult> {
        let mut result : Vec<TermApplicationResult> = Vec::new();
        for table in self.application_tables.values() {
            let mut temp = table.get_app_results_with_result(result_term).clone();
            result.append(&mut temp);
        }
        result
    }

    pub fn evaluate(self, term_app : &TermApplication) -> (InterpreterState, TermReference) {
        let func_type_id : TypeId = term_app.get_func_type();

        let application_table : &ApplicationTable = self.application_tables.get(&func_type_id).unwrap();

        if (application_table.has_computed(&term_app)) {
            let result : TermReference = application_table.get_computed(&term_app);
            (self, result)
        } else {
            let func_term : PartiallyAppliedTerm = {
                let func_space : &TypeSpace = self.type_spaces.get(&func_type_id).unwrap();
                func_space.get(term_app.func_ptr.index).clone()
            };
            let arg_ref : TermReference = term_app.arg_ref.clone();

            let func_impl = func_term.func_impl;
            let mut args_copy = func_term.args.clone();

            args_copy.push(arg_ref);

            let result_tuple : (InterpreterState, TermReference) = if (func_impl.ready_to_evaluate(&args_copy)) {
                func_impl.evaluate(self, args_copy)
            } else {
                let result = PartiallyAppliedTerm {
                    func_impl : func_impl,
                    args : args_copy
                };
                let ret_type_id : TypeId = term_app.get_ret_type();
                let (szelf, ret_ptr) = self.store_term(ret_type_id, result);
                let ret_ref = TermReference::FuncRef(ret_ptr);
                (szelf, ret_ref)
            };
            let (mut zelf, result_ref) = result_tuple;

            let application_table : &mut ApplicationTable = zelf.application_tables.get_mut(&func_type_id).unwrap();

            application_table.link(term_app.clone(), result_ref.clone());

            (zelf, result_ref)
        }
    }

    pub fn add_init(&mut self, func : EnumFuncImpl) -> TermPointer {
        let func_type_id : TypeId = func.func_type();
        let type_space : &mut TypeSpace = self.type_spaces.get_mut(&func_type_id).unwrap();
        type_space.add_init(func)
    }

    pub fn new() -> InterpreterState {
        //Initialize hashmaps for each type in the global type table 
        let mut application_tables = HashMap::<TypeId, ApplicationTable>::new();
        let mut type_spaces = HashMap::<TypeId, TypeSpace>::new();

        for i in 0..total_num_types() {
            let type_id : TypeId = i as TypeId;
            if let Type::FuncType(arg_type, ret_type) = get_type(type_id) {
                application_tables.insert(type_id, ApplicationTable::new(type_id));
                type_spaces.insert(type_id, TypeSpace::new(type_id));
            }
        }

        let mut result = InterpreterState {
            application_tables,
            type_spaces
        };

        //Now populate the type spaces with the known function implementations
        //using TypeSpace#add(PartiallyAppliedTerm)
        
        //Binary functions
        for type_id in [*SCALAR_T, *VECTOR_T].iter() {
            for op in [EnumBinaryArrayOperator::AddOperator(AddOperator {}),
                       EnumBinaryArrayOperator::SubOperator(SubOperator {}),
                       EnumBinaryArrayOperator::MulOperator(MulOperator {})].iter() {
                let func_impl = BinaryFuncImpl {
                    elem_type : *type_id,
                    f : op.clone()
                };
                result.add_init(EnumFuncImpl::BinaryFuncImpl(func_impl));
            }
        }

        result.add_init(EnumFuncImpl::MapImpl(MapImpl {}));
        result.add_init(EnumFuncImpl::FillImpl(FillImpl {}));
        result.add_init(EnumFuncImpl::SetHeadImpl(SetHeadImpl {}));
        result.add_init(EnumFuncImpl::HeadImpl(HeadImpl {}));
        result.add_init(EnumFuncImpl::RotateImpl(RotateImpl {}));
        result.add_init(EnumFuncImpl::ReduceImpl(ReduceImpl {}));

        //Constant functions
        for one_id in [*SCALAR_T, *VECTOR_T].iter() {
            for two_id in [*SCALAR_T, *VECTOR_T].iter() {
                let func_impl = ConstImpl {
                    ret_type : *one_id,
                    ignored_type : *two_id
                };
                result.add_init(EnumFuncImpl::ConstImpl(func_impl));
            }
        }
        
        //Function composition
        for one_id in [*SCALAR_T, *VECTOR_T].iter() {
            for two_id in [*SCALAR_T, *VECTOR_T].iter() {
                for three_id in [*SCALAR_T, *VECTOR_T].iter() {
                    let func_impl = ComposeImpl::new(*one_id, *two_id, *three_id);
                    result.add_init(EnumFuncImpl::ComposeImpl(func_impl));
                }
            }
        }
        result
    }
}
