CC ?= cc
CFLAGS ?= -O3 -D_FILE_OFFSET_BITS=64
LDFLAGS ?= -lm

.PHONY: build
build: glm4.7-flash.c
	($(CC) $(CFLAGS) -fopenmp -o glm4.7-flash glm4.7-flash.c $(LDFLAGS)) || ($(CC) $(CFLAGS) -o glm4.7-flash glm4.7-flash.c $(LDFLAGS))

.PHONY: export
export:
	python3 glm47_flash_export.py --input /Users/mj/glm.c/GLM-4.7-Flash-UD-Q6_K_XL.gguf --output /Users/mj/glm.c/GLM-4.7-Flash.bin

.PHONY: smoke
smoke: build
	./glm4.7-flash /Users/mj/glm.c/GLM-4.7-Flash.bin -n 16 -t 0 --seed 0 -i "1+1="

.PHONY: chat
chat: build
	./glm4.7-flash /Users/mj/glm.c/GLM-4.7-Flash.bin -m chat -n 64

.PHONY: smoke-tokenizer
smoke-tokenizer: build
	./glm4.7-flash /Users/mj/glm.c/GLM-4.7-Flash.bin --self-test-tokenizer

.PHONY: determinism
determinism: build
	./glm4.7-flash /Users/mj/glm.c/GLM-4.7-Flash.bin -n 16 -i "1+1=" > /tmp/glm_out_1.txt
	./glm4.7-flash /Users/mj/glm.c/GLM-4.7-Flash.bin -n 16 -i "1+1=" > /tmp/glm_out_2.txt
	diff -u /tmp/glm_out_1.txt /tmp/glm_out_2.txt

.PHONY: compare
compare:
	/opt/homebrew/bin/llama-cli -m /Users/mj/glm.c/GLM-4.7-Flash-UD-Q6_K_XL.gguf -ngl 0 -n 16 -p "1+1="

.PHONY: eval-callback
eval-callback: build
	@set -e; \
	prompt_file=/tmp/glm_eval_prompt.txt; \
	native_raw=/tmp/glm_eval_native.log; \
	ref_raw=/tmp/glm_eval_ref.log; \
	native_sum=/tmp/glm_eval_native_sums.txt; \
	ref_sum=/tmp/glm_eval_ref_sums.txt; \
	native_unsorted=/tmp/glm_eval_native_sums_unsorted.txt; \
	ref_unsorted=/tmp/glm_eval_ref_sums_unsorted.txt; \
	missing_native=/tmp/glm_eval_missing_native.txt; \
	missing_ref=/tmp/glm_eval_missing_ref.txt; \
	printf "Hello" > $$prompt_file; \
	echo "[glm4.7-flash debug dump]"; \
	./glm4.7-flash /Users/mj/glm.c/GLM-4.7-Flash.bin -m completion -n 1 --debug -f $$prompt_file > $$native_raw 2>&1; \
	sed -n '/^\[debug\] prompt_ids/p;/^\[glm-debug-callback\]/p;/^\[glm-layer-sum\]/p' $$native_raw; \
	echo "[llama-tokenize prompt ids]"; \
	/opt/homebrew/bin/llama-tokenize -m /Users/mj/glm.c/GLM-4.7-Flash-UD-Q6_K_XL.gguf -f $$prompt_file --ids --log-disable; \
	echo "[llama-eval-callback capture]"; \
	/opt/homebrew/bin/llama-eval-callback \
		-m /Users/mj/glm.c/GLM-4.7-Flash-UD-Q6_K_XL.gguf \
		-f $$prompt_file \
		-n 1 \
		-ngl 0 \
		--temp 0 \
		--seed 0 > $$ref_raw 2>&1; \
	awk '/^\[glm-layer-sum\] /{name=$$2; sum=$$3; sub(/^sum=/, "", sum); print name, sum;}' $$native_raw > $$native_unsorted; \
	sort -k1,1 $$native_unsorted > $$native_sum; \
	awk '\
		/^common_debug_cb_eval:/ { \
			line=$$0; \
			sub(/^common_debug_cb_eval:[[:space:]]*/, "", line); \
			split(line, a, " = "); \
			cur=a[1]; \
			gsub(/^[[:space:]]+|[[:space:]]+$$/, "", cur); \
			next; \
		} \
		cur != "" && index($$0, "sum = ") > 0 { \
			sum = $$0; \
			sub(/^.*sum = /, "", sum); \
			sub(/[[:space:]].*$$/, "", sum); \
			if (cur == "embd" || cur == "result_norm" || cur == "result_output" || cur ~ /^l_out-[0-9]+$$/) print cur, sum; \
			cur = ""; \
		} \
	' $$ref_raw > $$ref_unsorted; \
	sort -k1,1 $$ref_unsorted > $$ref_sum; \
	echo "[layer-by-layer sum compare]"; \
	printf "%-14s %14s %14s %12s\n" tensor glm_sum llama_sum abs_diff; \
	join -j 1 $$native_sum $$ref_sum | awk '{glm=$$2+0; ref=$$3+0; d=glm-ref; if (d < 0) d = -d; printf "%-14s %14.6f %14.6f %12.6f\n", $$1, glm, ref, d;}'; \
	join -v 1 $$native_sum $$ref_sum > $$missing_ref; \
	join -v 2 $$native_sum $$ref_sum > $$missing_native; \
	if [ -s $$missing_ref ]; then echo "[missing in llama-eval-callback]"; awk '{print "  " $$0}' $$missing_ref; fi; \
	if [ -s $$missing_native ]; then echo "[missing in glm4.7-flash]"; awk '{print "  " $$0}' $$missing_native; fi; \
	echo "[artifacts] $$native_raw $$ref_raw $$native_sum $$ref_sum"

.PHONY: probe-coherence
probe-coherence: build
	@out=/tmp/glm_probe_latest.txt; \
	rm -f "$$out"; \
	echo "# GLM native coherence probe" >> "$$out"; \
	echo "# date: $$(date -u '+%Y-%m-%dT%H:%M:%SZ')" >> "$$out"; \
	echo "# bin: /Users/mj/glm.c/GLM-4.7-Flash.bin" >> "$$out"; \
	echo "" >> "$$out"; \
	for p in "1+1=" "31+14=" "41-16=" "50-8=" "9/3=" "Earth has" "The chemical symbol for water is"; do \
		echo "## prompt: $$p" >> "$$out"; \
		./glm4.7-flash /Users/mj/glm.c/GLM-4.7-Flash.bin -n 8 -i "$$p" >> "$$out"; \
		echo "" >> "$$out"; \
	done; \
	echo "Wrote $$out"; \
	tail -n 80 "$$out"

.PHONY: probe-coherence-core
probe-coherence-core: build
	@out=/tmp/glm_probe_core_latest.txt; \
	rm -f "$$out"; \
	echo "# GLM native core coherence probe" >> "$$out"; \
	echo "# date: $$(date -u '+%Y-%m-%dT%H:%M:%SZ')" >> "$$out"; \
	echo "" >> "$$out"; \
	for p in "1+1=" "31+14=" "41-16="; do \
		echo "## prompt: $$p" >> "$$out"; \
		./glm4.7-flash /Users/mj/glm.c/GLM-4.7-Flash.bin -n 8 -i "$$p" >> "$$out"; \
		echo "" >> "$$out"; \
	done; \
	echo "Wrote $$out"; \
	cat "$$out"

.PHONY: probe-parity-core
probe-parity-core: build
	python3 /Users/mj/glm.c/probe_parity_core.py \
		--native-bin /Users/mj/glm.c/glm4.7-flash \
		--native-model /Users/mj/glm.c/GLM-4.7-Flash.bin \
		--ref-bin /opt/homebrew/bin/llama-completion \
		--ref-model /Users/mj/glm.c/GLM-4.7-Flash-UD-Q6_K_XL.gguf \
		--tokens 8 \
		--output /tmp/glm_parity_core_latest.txt
