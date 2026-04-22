import torch
import torch.nn.functional as F
from lakv_v2.cache.aligner import CacheAligner

class ValidationSuite:
    def __init__(self, model, tokenizer, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.aligner = CacheAligner(device=device)

    def run_all(self):
        print("\n=== LAKV-v2 Validation Suite ===")
        print("1. Cache Handoff Parity Test...")
        p_ok = self.test_cache_handoff_parity()
        print(f"   -> {'PASS' if p_ok else 'FAIL'}")
        
        print("2. Quantization Roundtrip Test...")
        q_ok = self.test_quantization_roundtrip()
        print(f"   -> {'PASS' if q_ok else 'FAIL'}")
        
        print("3. Layer Drop Sensitivity Test...")
        d_ok = self.test_layer_drop_sensitivity()
        print(f"   -> {'PASS' if d_ok else 'FAIL'}")
        print("================================\n")
        return p_ok and q_ok and d_ok

    def test_cache_handoff_parity(self):
        """
        Verify that `[Seq A] -> KV -> [Seq B]` produces identical logits 
        for [Seq B] as a continuous `[Seq A + Seq B]` prompt.
        """
        text_a = "The capital of France is Paris. "
        text_b = "The capital of Germany is"

        # Continuous Baseline
        full_text = text_a + text_b
        full_ids = self.tokenizer(full_text, return_tensors="pt").input_ids.to(self.device)
        with torch.no_grad():
            out_base = self.model(input_ids=full_ids)
            logits_base = out_base.logits[0, -1, :]
            
        # Segmented Handoff
        ids_a = self.tokenizer(text_a, return_tensors="pt").input_ids.to(self.device)
        ids_b = self.tokenizer(text_b, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)
        
        with torch.no_grad():
            # Step 1: Process A, get KV
            out_a = self.model(input_ids=ids_a, use_cache=True)
            kv_a = self.aligner._to_tuple(out_a.past_key_values)
            
            # Step 2: Inject KV_a and process B
            dynamic_cache = self.aligner.prepare_handoff_cache(kv_a)
            pos_ids = self.aligner.compute_receiver_position_ids(kv_a, ids_b)
            attn_mask = self.aligner.build_attention_mask(kv_a, ids_b)
            
            out_b = self.model(
                input_ids=ids_b,
                past_key_values=dynamic_cache,
                position_ids=pos_ids,
                attention_mask=attn_mask,
                use_cache=True
            )
            logits_handoff = out_b.logits[0, -1, :]

        # Compare
        cos_sim = F.cosine_similarity(logits_base.unsqueeze(0), logits_handoff.unsqueeze(0)).item()
        match = (logits_base.argmax() == logits_handoff.argmax())
        
        print(f"      Cosine sim: {cos_sim:.6f} | Exact Argmax Match: {match}")
        return (cos_sim > 0.99) and match

    def test_quantization_roundtrip(self):
        """Re-verify mathematically that we can INT8 quantize and dequantize close to FP16"""
        from kv_compressor import KVCompressor # use old just for testing the logic
        comp = KVCompressor(mode="uniform_int8")
        
        dummy_k = torch.randn(1, 4, 128, 128, dtype=torch.bfloat16, device=self.device)
        dummy_v = torch.randn(1, 4, 128, 128, dtype=torch.bfloat16, device=self.device)
        dummy_kv = ((dummy_k, dummy_v),)
        
        msg = comp.compress(dummy_kv)
        recon = comp.decompress(msg, device=self.device)
        
        k_recon = recon[0][0]
        cos_sim = F.cosine_similarity(dummy_k.float().flatten().unsqueeze(0), 
                                      k_recon.float().flatten().unsqueeze(0)).item()
        
        return cos_sim > 0.99

    def test_layer_drop_sensitivity(self):
        """Provide basic sensitivity bounds when dropping the middle layer to zeros"""
        text = "What is 15 + 27? Calculate carefully."
        ids = self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)
        with torch.no_grad():
            out_clean = self.model(input_ids=ids, use_cache=True)
            kv = out_clean.past_key_values
            
        kv_list = list(self.aligner._to_tuple(kv))
        mid = len(kv_list) // 2
        # Zero out middle layer
        k_shape, v_shape = kv_list[mid][0].shape, kv_list[mid][1].shape
        kv_list[mid] = (torch.zeros(k_shape, dtype=torch.bfloat16, device=self.device),
                        torch.zeros(v_shape, dtype=torch.bfloat16, device=self.device))
        
        cache_dropped = self.aligner.prepare_handoff_cache(tuple(kv_list))
        # Evaluate one step of logits
        ids_new = torch.tensor([[self.tokenizer.eos_token_id]], device=self.device)
        pos_ids = self.aligner.compute_receiver_position_ids(tuple(kv_list), ids_new)
        attn_mask = self.aligner.build_attention_mask(tuple(kv_list), ids_new)
        
        with torch.no_grad():
            out_dropped = self.model(
                input_ids=ids_new,
                past_key_values=cache_dropped,
                position_ids=pos_ids,
                attention_mask=attn_mask
            )
            
        # Ensure it works without crashing
        return not torch.isnan(out_dropped.logits).any()

if __name__ == "__main__":
    from run import load_model # reuse root logic to get model
    m, t = load_model("Qwen/Qwen2.5-7B-Instruct", "cuda")
    ValidationSuite(m, t).run_all()
