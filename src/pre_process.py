def create_meta_variables(df):
    """
    Create meta-variables by grouping building typologies into broader categories.
    
    Parameters:
    df (pandas.DataFrame): DataFrame containing the building typology percentage columns
    
    Returns:
    pandas.DataFrame: DataFrame with original columns plus new meta-variables
    """
    
    # Create a copy to avoid modifying original data
    df_meta = df.copy()
    
    # Standard sized properties (standard detached + standard semi)
    standard_cols = ['Standard size detached_pct', 'Standard size semi detached_pct']
    df_meta['standard_sized_pct'] = df_meta[standard_cols].sum(axis=1)
    
    # Large properties (large detached + large semi + very large detached)
    large_cols = ['Large detached_pct', 'Large semi detached_pct', 'Very large detached_pct']
    df_meta['large_properties_pct'] = df_meta[large_cols].sum(axis=1)
    
    # All flats (combining all flat categories)
    flat_cols = ['3-4 storey and smaller flats_pct', 'Medium height flats 5-6 storeys_pct', 
                 'Tall flats 6-15 storeys_pct', 'Very tall point block flats_pct']
    df_meta['all_flats_pct'] = df_meta[flat_cols].sum(axis=1)
    
    # Terraces and mixed (terraces + mixed estates + linked premises)
    terraces_mixed_cols = ['2 storeys terraces with t rear extension_pct', 'Small low terraces_pct',
                          'Tall terraces 3-4 storeys_pct', 'Planned balanced mixed estates_pct',
                          'Linked and step linked premises_pct']
    df_meta['terraces_mixed_pct'] = df_meta[terraces_mixed_cols].sum(axis=1)
    
    # All detached (standard + large + very large)
    detached_cols = ['Standard size detached_pct', 'Large detached_pct', 'Very large detached_pct']
    df_meta['all_detached_pct'] = df_meta[detached_cols].sum(axis=1)
    
    # All semi-detached (standard + large)
    semi_cols = ['Standard size semi detached_pct', 'Large semi detached_pct']
    df_meta['all_semi_detached_pct'] = df_meta[semi_cols].sum(axis=1)
    
    # High-density housing (all flats + tall terraces + multi-unit)
    high_density_cols = ['3-4 storey and smaller flats_pct', 'Medium height flats 5-6 storeys_pct',
                        'Tall flats 6-15 storeys_pct', 'Very tall point block flats_pct',
                        'Tall terraces 3-4 storeys_pct', 'Semi type house in multiples_pct']
    df_meta['high_density_pct'] = df_meta[high_density_cols].sum(axis=1)
    
    # Low-density housing (all detached + all semi-detached)
    low_density_cols = ['Standard size detached_pct', 'Large detached_pct', 'Very large detached_pct',
                       'Standard size semi detached_pct', 'Large semi detached_pct']
    df_meta['low_density_pct'] = df_meta[low_density_cols].sum(axis=1)
    
     # All typologies (total of all building types)
    all_typology_cols = ['Standard size detached_pct', 'Large detached_pct', 'Very large detached_pct',
                        'Standard size semi detached_pct', 'Large semi detached_pct',
                        '3-4 storey and smaller flats_pct', 'Medium height flats 5-6 storeys_pct',
                        'Tall flats 6-15 storeys_pct', 'Very tall point block flats_pct',
                        '2 storeys terraces with t rear extension_pct', 'Small low terraces_pct',
                        'Tall terraces 3-4 storeys_pct', 'Planned balanced mixed estates_pct',
                        'Linked and step linked premises_pct', 'Semi type house in multiples_pct']
    df_meta['all_typologies_pct'] = df_meta[all_typology_cols].sum(axis=1)
    
    return df_meta