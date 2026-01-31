from datasets import load_dataset

def audit_dmr_structure():
    print("--- Loading DMR-IR Metadata ---")
    # Load in streaming mode just to peek at metadata quickly
    ds = load_dataset("SemilleroCV/DMR-IR", split="train", streaming=True)
    
    print("\n[Available Features/Columns]:")
    print(ds.features)
    
    print("\n[Inspecting First 10 Sample IDs for View Codes]:")
    
    # We look at the 'id' or filename to see if it says 'Front', 'Lat', etc.
    ids = []
    for i, sample in enumerate(ds):
        if i >= 10: break
        # Try to find an ID field (it might be 'file_name', 'id', 'patient_id')
        # We print all keys for the first sample to be sure
        if i == 0:
            print(f"Sample Keys: {sample.keys()}")
        
        # Look for typical ID keys
        sample_id = sample.get('id', sample.get('file_name', 'Unknown'))
        ids.append(sample_id)
        
    print(ids)

if __name__ == "__main__":
    audit_dmr_structure()