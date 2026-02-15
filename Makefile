CC ?= cc
CFLAGS ?= -O3 -D_FILE_OFFSET_BITS=64
LDFLAGS ?= -lm
UNAME_S := $(shell uname -s)
PROJECT_ROOT := $(abspath .)
OPENMP_PREFIX ?= /opt/homebrew/opt/libomp
OPENMP_MODE ?= $(shell tmpc=/tmp/glm_openmp_probe_$$.c; tmpx=/tmp/glm_openmp_probe_$$.bin; \
	printf 'int main(void){return 0;}\n' > $$tmpc; \
	if $(CC) -fopenmp $$tmpc -o $$tmpx >/dev/null 2>&1; then \
		echo native; \
	elif $(CC) -Xpreprocessor -fopenmp -I$(OPENMP_PREFIX)/include $$tmpc -L$(OPENMP_PREFIX)/lib -lomp -o $$tmpx >/dev/null 2>&1; then \
		echo apple-libomp; \
	else \
		echo none; \
	fi; \
	rm -f $$tmpc $$tmpx)

ifeq ($(OPENMP_MODE),native)
OPENMP_CFLAGS ?= -fopenmp
OPENMP_LDFLAGS ?= -fopenmp
else ifeq ($(OPENMP_MODE),apple-libomp)
OPENMP_CFLAGS ?= -Xpreprocessor -fopenmp -I$(OPENMP_PREFIX)/include
OPENMP_LDFLAGS ?= -L$(OPENMP_PREFIX)/lib -lomp
else
OPENMP_CFLAGS ?=
OPENMP_LDFLAGS ?=
endif

NATIVE_MODEL ?= $(PROJECT_ROOT)/GLM-4.7-Flash.bin
GGUF_MODEL ?= $(PROJECT_ROOT)/GLM-4.7-Flash-Q4_K_M.gguf
REF_MODEL ?= $(GGUF_MODEL)

LLAMA_CLI ?= /opt/homebrew/bin/llama-cli
LLAMA_TOKENIZE ?= /opt/homebrew/bin/llama-tokenize
LLAMA_EVAL_CB ?= /opt/homebrew/bin/llama-eval-callback
LLAMA_COMPLETION ?= /opt/homebrew/bin/llama-completion
LLAMA_BENCH ?= /opt/homebrew/bin/llama-bench
EVAL_BACKEND ?= cpu
EVAL_REF_NGL ?= 0
BENCH_TOKENS ?= 16
BENCH_PROMPT ?= 1+1=
BENCH_WARMUP ?= 1
BENCH_OMP_THREADS ?= 8
BENCH_OMP_THREAD_LIST ?= 1 2 4 8
BENCH_OMP_PROC_BIND ?= true
BENCH_OMP_PLACES ?= cores

.PHONY: build
build: glm4.7-flash.c
	$(CC) $(CFLAGS) $(OPENMP_CFLAGS) -o glm4.7-flash glm4.7-flash.c $(LDFLAGS) $(OPENMP_LDFLAGS)

.PHONY: openmp-info
openmp-info:
	@echo "OPENMP_MODE=$(OPENMP_MODE)"
	@echo "OPENMP_CFLAGS=$(OPENMP_CFLAGS)"
	@echo "OPENMP_LDFLAGS=$(OPENMP_LDFLAGS)"
	@if [ "$(OPENMP_MODE)" = "none" ]; then \
		echo "OpenMP is not active for CC='$(CC)'"; \
		echo "macOS quick fix: brew install libomp"; \
		echo "Then run: make build CC=cc"; \
	fi

ifeq ($(UNAME_S),Darwin)
.PHONY: metal-prereqs
metal-prereqs:
	@echo "Metal build prerequisites:"
	@echo "  xcode-select --install"
	@echo "  sudo xcode-select -s /Applications/Xcode.app/Contents/Developer"
	@echo "  sudo xcodebuild -license accept"
	@echo "  sudo xcodebuild -downloadComponent MetalToolchain"

.PHONY: check-metal-toolchain
check-metal-toolchain:
	@xcrun -sdk macosx --find metal >/dev/null 2>&1 || \
		(echo "missing Metal toolchain; run 'make metal-prereqs' and retry" && exit 1)

.PHONY: metal-lib
metal-lib: check-metal-toolchain glm4.7-flash.metal
	xcrun -sdk macosx metal -std=metal3.0 -O0 -c glm4.7-flash.metal -o glm4.7-flash.air
	xcrun -sdk macosx metallib glm4.7-flash.air -o glm4.7-flash.metallib

.PHONY: build-metal
build-metal: metal-lib glm4.7-flash.c glm4.7-flash.m
	$(CC) $(CFLAGS) $(OPENMP_CFLAGS) -DGLM_ENABLE_METAL=1 -x objective-c -fobjc-arc -o glm4.7-flash glm4.7-flash.c glm4.7-flash.m -framework Foundation -framework Metal $(LDFLAGS) $(OPENMP_LDFLAGS)
else
.PHONY: metal-prereqs
metal-prereqs:
	@echo "metal-prereqs is only available on Apple platforms"
	@exit 1

.PHONY: metal-lib
metal-lib:
	@echo "metal-lib is only available on Apple platforms"
	@exit 1

.PHONY: build-metal
build-metal:
	@echo "build-metal is only available on Apple platforms"
	@exit 1
endif

.PHONY: export
export:
	python3 glm47_flash_export.py --input "$(GGUF_MODEL)" --output "$(NATIVE_MODEL)"

.PHONY: smoke
smoke: build
	./glm4.7-flash "$(NATIVE_MODEL)" -n 16 -t 0 --seed 0 -i "1+1="

.PHONY: chat
chat: build
	./glm4.7-flash "$(NATIVE_MODEL)" -m chat -n 64

.PHONY: smoke-tokenizer
smoke-tokenizer: build
	./glm4.7-flash "$(NATIVE_MODEL)" --self-test-tokenizer

.PHONY: determinism
determinism: build
	./glm4.7-flash "$(NATIVE_MODEL)" -n 16 -i "1+1=" > /tmp/glm_out_1.txt
	./glm4.7-flash "$(NATIVE_MODEL)" -n 16 -i "1+1=" > /tmp/glm_out_2.txt
	diff -u /tmp/glm_out_1.txt /tmp/glm_out_2.txt

.PHONY: parity-checklist
parity-checklist: build-metal
	./glm4.7-flash "$(NATIVE_MODEL)" --backend cpu -n 16 -t 0 --seed 0 -i "1+1=" > /tmp/glm_cpu.txt
	./glm4.7-flash "$(NATIVE_MODEL)" --backend metal -n 16 -t 0 --seed 0 -i "1+1=" > /tmp/glm_metal.txt
	diff -u -I '^  backend=' /tmp/glm_cpu.txt /tmp/glm_metal.txt

.PHONY: compare
compare:
	$(LLAMA_CLI) -m "$(REF_MODEL)" -ngl 0 -n 16 -p "1+1="

.PHONY: eval-callback-run
eval-callback-run:
	@set -e; \
	backend="$(EVAL_BACKEND)"; \
	prompt_file=/tmp/glm_eval_prompt_$${backend}.txt; \
	native_raw=/tmp/glm_eval_native_$${backend}.log; \
	ref_raw=/tmp/glm_eval_ref_$${backend}.log; \
	native_sum=/tmp/glm_eval_native_sums_$${backend}.txt; \
	ref_sum=/tmp/glm_eval_ref_sums_$${backend}.txt; \
	native_unsorted=/tmp/glm_eval_native_sums_unsorted_$${backend}.txt; \
	ref_unsorted=/tmp/glm_eval_ref_sums_unsorted_$${backend}.txt; \
	missing_native=/tmp/glm_eval_missing_native_$${backend}.txt; \
	missing_ref=/tmp/glm_eval_missing_ref_$${backend}.txt; \
	printf "Hello" > $$prompt_file; \
	echo "[glm4.7-flash debug dump backend=$$backend]"; \
	./glm4.7-flash "$(NATIVE_MODEL)" --backend $$backend -m completion -n 1 --seed 0 --debug -f $$prompt_file > $$native_raw 2>&1; \
	sed -n '/^\[debug\] prompt_ids/p;/^\[glm-debug-callback\]/p;/^\[glm-layer-sum\]/p' $$native_raw; \
	echo "[llama-tokenize prompt ids]"; \
	$(LLAMA_TOKENIZE) -m "$(REF_MODEL)" -f $$prompt_file --ids --log-disable; \
	echo "[llama-eval-callback capture ngl=$(EVAL_REF_NGL)]"; \
	$(LLAMA_EVAL_CB) \
		-m "$(REF_MODEL)" \
		-f $$prompt_file \
		-n 1 \
		-ngl $(EVAL_REF_NGL) \
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

.PHONY: eval-callback eval-callback-cpu eval-callback-metal
eval-callback: eval-callback-cpu

eval-callback-cpu: build
	@$(MAKE) eval-callback-run EVAL_BACKEND=cpu EVAL_REF_NGL=0

eval-callback-metal: build-metal
	@$(MAKE) eval-callback-run EVAL_BACKEND=metal EVAL_REF_NGL=99

.PHONY: probe-coherence
probe-coherence: build
	@out=/tmp/glm_probe_latest.txt; \
	rm -f "$$out"; \
	echo "# GLM native coherence probe" >> "$$out"; \
	echo "# date: $$(date -u '+%Y-%m-%dT%H:%M:%SZ')" >> "$$out"; \
	echo "# bin: $(NATIVE_MODEL)" >> "$$out"; \
	echo "" >> "$$out"; \
	for p in "1+1=" "31+14=" "41-16=" "50-8=" "9/3=" "Earth has" "The chemical symbol for water is"; do \
		echo "## prompt: $$p" >> "$$out"; \
		./glm4.7-flash "$(NATIVE_MODEL)" -n 8 -i "$$p" >> "$$out"; \
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
		./glm4.7-flash "$(NATIVE_MODEL)" -n 8 -i "$$p" >> "$$out"; \
		echo "" >> "$$out"; \
	done; \
	echo "Wrote $$out"; \
	cat "$$out"

.PHONY: probe-parity-core
probe-parity-core: build
	python3 "$(PROJECT_ROOT)/probe_parity_core.py" \
		--native-bin "$(PROJECT_ROOT)/glm4.7-flash" \
		--native-model "$(NATIVE_MODEL)" \
		--ref-bin "$(LLAMA_COMPLETION)" \
		--ref-model "$(REF_MODEL)" \
		--tokens 8 \
		--output /tmp/glm_parity_core_latest.txt

.PHONY: bench-tps
bench-tps: build-metal
	@set -e; \
	tokens="$(BENCH_TOKENS)"; \
	prompt="$(BENCH_PROMPT)"; \
	glm_cpu_out=/tmp/glm_tps_glm-cpu.txt; \
	glm_cpu_time=/tmp/glm_tps_glm-cpu.time.txt; \
	glm_metal_out=/tmp/glm_tps_glm-metal.txt; \
	glm_metal_time=/tmp/glm_tps_glm-metal.time.txt; \
	llama_cpu_json=/tmp/glm_tps_llama-bench-cpu.json; \
	llama_metal_json=/tmp/glm_tps_llama-bench-metal.json; \
	if [ "$(BENCH_WARMUP)" = "1" ]; then \
		echo "[bench] warmup glm.c cpu"; \
		env OMP_NUM_THREADS=$(BENCH_OMP_THREADS) OMP_PROC_BIND=$(BENCH_OMP_PROC_BIND) OMP_PLACES=$(BENCH_OMP_PLACES) \
			./glm4.7-flash "$(NATIVE_MODEL)" --backend cpu -n 1 -t 0 --seed 0 -i "$$prompt" > /dev/null; \
		echo "[bench] warmup glm.c metal"; \
		./glm4.7-flash "$(NATIVE_MODEL)" --backend metal -n 1 -t 0 --seed 0 -i "$$prompt" > /dev/null; \
	fi; \
	echo "[bench] glm.c cpu"; \
	env OMP_NUM_THREADS=$(BENCH_OMP_THREADS) OMP_PROC_BIND=$(BENCH_OMP_PROC_BIND) OMP_PLACES=$(BENCH_OMP_PLACES) \
		/usr/bin/time -p ./glm4.7-flash "$(NATIVE_MODEL)" --backend cpu -n $$tokens -t 0 --seed 0 -i "$$prompt" > $$glm_cpu_out 2> $$glm_cpu_time; \
	echo "[bench] glm.c metal"; \
	/usr/bin/time -p ./glm4.7-flash "$(NATIVE_MODEL)" --backend metal -n $$tokens -t 0 --seed 0 -i "$$prompt" > $$glm_metal_out 2> $$glm_metal_time; \
	echo "[bench] llama.cpp cpu"; \
	$(LLAMA_BENCH) -m "$(REF_MODEL)" -ngl 0 -p 1 -n $$tokens -r 1 -o json > $$llama_cpu_json; \
	echo "[bench] llama.cpp metal"; \
	$(LLAMA_BENCH) -m "$(REF_MODEL)" -ngl 99 -p 1 -n $$tokens -r 1 -o json > $$llama_metal_json; \
	glm_cpu_tps=$$(awk -v n=$$tokens '/^real /{if ($$2 > 0) printf "%.3f", n/$$2}' $$glm_cpu_time); \
	glm_metal_tps=$$(awk -v n=$$tokens '/^real /{if ($$2 > 0) printf "%.3f", n/$$2}' $$glm_metal_time); \
	llama_cpu_tps=$$(F=$$llama_cpu_json python3 -c "import json, os; d=json.load(open(os.environ['F'])); g=next((x for x in d if x.get('n_gen', 0) > 0), None); print('%.3f' % g['avg_ts'] if g else 'nan')"); \
	llama_metal_tps=$$(F=$$llama_metal_json python3 -c "import json, os; d=json.load(open(os.environ['F'])); g=next((x for x in d if x.get('n_gen', 0) > 0), None); print('%.3f' % g['avg_ts'] if g else 'nan')"); \
	echo ""; \
	echo "TPS benchmark (tokens=$$tokens, prompt='$$prompt')"; \
	printf "%-22s %8s\n" "mode" "tok/s"; \
	printf "%-22s %8s\n" "glm.c cpu (omp=$(BENCH_OMP_THREADS))" "$$glm_cpu_tps"; \
	printf "%-22s %8s\n" "glm.c metal" "$$glm_metal_tps"; \
	printf "%-22s %8s\n" "llama.cpp cpu" "$$llama_cpu_tps"; \
	printf "%-22s %8s\n" "llama.cpp metal" "$$llama_metal_tps"; \
	echo "[artifacts] $$glm_cpu_out $$glm_cpu_time $$glm_metal_out $$glm_metal_time $$llama_cpu_json $$llama_metal_json"

.PHONY: bench-tps-omp
bench-tps-omp: build
	@set -e; \
	tokens="$(BENCH_TOKENS)"; \
	prompt="$(BENCH_PROMPT)"; \
	echo "[bench] OpenMP thread sweep (tokens=$$tokens, prompt='$$prompt')"; \
	printf "%-8s %8s\n" "threads" "tok/s"; \
	for t in $(BENCH_OMP_THREAD_LIST); do \
		out=/tmp/glm_tps_glm-cpu-omp-t$$t.txt; \
		timef=/tmp/glm_tps_glm-cpu-omp-t$$t.time.txt; \
		if [ "$(BENCH_WARMUP)" = "1" ]; then \
			env OMP_NUM_THREADS=$$t OMP_PROC_BIND=$(BENCH_OMP_PROC_BIND) OMP_PLACES=$(BENCH_OMP_PLACES) \
				./glm4.7-flash "$(NATIVE_MODEL)" --backend cpu -n 1 -t 0 --seed 0 -i "$$prompt" > /dev/null; \
		fi; \
		env OMP_NUM_THREADS=$$t OMP_PROC_BIND=$(BENCH_OMP_PROC_BIND) OMP_PLACES=$(BENCH_OMP_PLACES) \
			/usr/bin/time -p ./glm4.7-flash "$(NATIVE_MODEL)" --backend cpu -n $$tokens -t 0 --seed 0 -i "$$prompt" > $$out 2> $$timef; \
		tps=$$(awk -v n=$$tokens '/^real /{if ($$2 > 0) printf "%.3f", n/$$2}' $$timef); \
		printf "%-8s %8s\n" $$t $$tps; \
	done; \
	echo "[artifacts] /tmp/glm_tps_glm-cpu-omp-t*.txt /tmp/glm_tps_glm-cpu-omp-t*.time.txt"
