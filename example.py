from customizable_synthesizer import CuTS
import pandas as pd

program = '''
SYNTHESIZE: Adult;

    ENFORCE: STATISTICAL:  
        E[age|age > 30] == 40;
    
END;
'''    

cuts = CuTS(program)
cuts.fit(verbose=True)

syndata = cuts.generate_data(30000)

# Debug: Print dimensions
print(f"Syndata shape: {syndata.shape}")
print(f"Dataset mean shape: {cuts.dataset.mean.shape}")
print(f"Dataset std shape: {cuts.dataset.std.shape}")
print(f"Number of train features: {len(cuts.dataset.train_features)}")
print(f"Number of all features: {len(cuts.dataset.features)}")

# Decode the full one-hot encoded data back to original format
# NOTE: CUTS trains on FULL one-hot encoded data (with discretized continuous features)
# So we need to use decode_full_one_hot_batch, not decode_batch
decoded_data = cuts.dataset.decode_full_one_hot_batch(
    syndata, 
    buckets=32, 
    with_label=True, 
    input_torch=True
)
syndata_df = pd.DataFrame(decoded_data, columns=cuts.dataset.features.keys())
syndata_df.to_csv('synthetic_adult_data.csv', index=False)
print(f"Synthetic data saved to synthetic_adult_data.csv with shape: {syndata_df.shape}")