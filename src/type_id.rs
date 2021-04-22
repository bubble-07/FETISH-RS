use fetish_lib::everything::*;

pub fn get_default_type_info_directory(dim : usize) -> TypeInfoDirectory {
    let mut types : TypeInfoDirectory = TypeInfoDirectory::new();
    let scalar_t = types.add(Type::VecType(1));
    let vector_t = types.add(Type::VecType(dim));
    let unary_scalar_func_t = types.add(Type::FuncType(scalar_t, scalar_t));
    let unary_vec_func_t = types.add(Type::FuncType(vector_t, vector_t));
    let _binary_vec_func_t = types.add(Type::FuncType(vector_t, unary_vec_func_t));
    let binary_scalar_func_t = types.add(Type::FuncType(scalar_t, unary_scalar_func_t));
    let _map_func_t = types.add(Type::FuncType(unary_scalar_func_t, unary_vec_func_t));
    let vector_to_scalar_func_t = types.add(Type::FuncType(vector_t, scalar_t));
    let reduce_temp_t = types.add(Type::FuncType(scalar_t, vector_to_scalar_func_t));
    let _reduce_func_t = types.add(Type::FuncType(binary_scalar_func_t, reduce_temp_t));
    let _fill_func_t = types.add(Type::FuncType(scalar_t, vector_t));
    let scalar_to_vector_func_t = types.add(Type::FuncType(scalar_t, vector_t));
    let _set_head_func_t = types.add(Type::FuncType(vector_t, scalar_to_vector_func_t));
    
    info!("Adding composition types");
    
    //Add all composition types of vector functions
    for n_t in [scalar_t, vector_t].iter() {
        for m_t in [scalar_t, vector_t].iter() {
            for p_t in [scalar_t, vector_t].iter() {
                let func_one = types.add(Type::FuncType(*m_t, *p_t));
                let func_two = types.add(Type::FuncType(*n_t, *m_t));
                let func_out = types.add(Type::FuncType(*n_t, *p_t));

                let two_to_out = types.add(Type::FuncType(func_two, func_out));
                let _compose_type = types.add(Type::FuncType(func_one, two_to_out));
            }
        }
    }

    info!("Adding constant types");
    //Add in all constant functions
    for n_t in [scalar_t, vector_t].iter() {
        for m_t in [scalar_t, vector_t].iter() {
            let out_func_t = types.add(Type::FuncType(*m_t, *n_t));
            let _const_func_t = types.add(Type::FuncType(*n_t, out_func_t));
        }
    }
    
    info!("Type initialization complete");

    types
}
