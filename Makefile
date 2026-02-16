CC ?= cc
CFLAGS ?= -O3 -D_FILE_OFFSET_BITS=64
LDFLAGS ?= -lm
UNAME_S := $(shell uname -s)
PROJECT_ROOT := $(abspath .)
MODEL_DIR ?= $(PROJECT_ROOT)/model
BIN ?= glm4.7-flash
METAL_SRC ?= infer.metal
METAL_AIR ?= infer.air
METAL_LIB ?= infer.metallib
OPENMP_PREFIX ?= /opt/homebrew/opt/libomp
ifndef OPENMP_MODE
OPENMP_MODE := $(shell tmpc=/tmp/glm_openmp_probe_$$.c; tmpx=/tmp/glm_openmp_probe_$$.bin; \
	printf 'int main(void){return 0;}\n' > $$tmpc; \
	if $(CC) -fopenmp $$tmpc -o $$tmpx >/dev/null 2>&1; then \
		echo native; \
	elif $(CC) -Xpreprocessor -fopenmp -I$(OPENMP_PREFIX)/include $$tmpc -L$(OPENMP_PREFIX)/lib -lomp -o $$tmpx >/dev/null 2>&1; then \
		echo apple-libomp; \
	else \
		echo none; \
	fi; \
	rm -f $$tmpc $$tmpx)
endif

ifeq ($(OPENMP_MODE),native)
OPENMP_CFLAGS := -fopenmp
OPENMP_LDFLAGS := -fopenmp
else ifeq ($(OPENMP_MODE),apple-libomp)
OPENMP_CFLAGS := -Xpreprocessor -fopenmp -I$(OPENMP_PREFIX)/include
OPENMP_LDFLAGS := -L$(OPENMP_PREFIX)/lib -lomp
else
OPENMP_CFLAGS :=
OPENMP_LDFLAGS :=
endif

BLAS ?= auto
ifeq ($(BLAS),auto)
ifeq ($(UNAME_S),Darwin)
BLAS_MODE := accelerate
else
BLAS_MODE := $(shell pkg-config --exists openblas && echo openblas || echo none)
endif
else
BLAS_MODE := $(BLAS)
endif

ifeq ($(BLAS_MODE),accelerate)
BLAS_CFLAGS := -DGLM_USE_ACCELERATE -DACCELERATE_NEW_LAPACK
BLAS_LDFLAGS := -framework Accelerate
else ifeq ($(BLAS_MODE),openblas)
BLAS_CFLAGS := -DGLM_USE_CBLAS $(shell pkg-config --cflags openblas 2>/dev/null)
BLAS_LDFLAGS := $(shell pkg-config --libs openblas 2>/dev/null)
ifeq ($(strip $(BLAS_LDFLAGS)),)
BLAS_LDFLAGS := -lopenblas
endif
else
BLAS_CFLAGS :=
BLAS_LDFLAGS :=
endif

NATIVE_MODEL ?= $(MODEL_DIR)/GLM-4.7-Flash.bin
GGUF_MODEL ?= $(MODEL_DIR)/GLM-4.7-Flash-Q4_K_M.gguf
REF_MODEL ?= $(GGUF_MODEL)

APP_SOURCES := main.c infer.c backend.c
METAL_SOURCES := $(APP_SOURCES) infer.m

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
BENCH_PREFILL_PROMPT ?= The quick brown fox jumps over the lazy dog. Repeat this sentence for prefill benchmarking.
CAPTURE_PROMPT ?= 1+1=
METAL_NATIVE ?= 0

.PHONY: build
build: $(APP_SOURCES)
	$(CC) $(CFLAGS) $(OPENMP_CFLAGS) $(BLAS_CFLAGS) -o $(BIN) $(APP_SOURCES) $(LDFLAGS) $(OPENMP_LDFLAGS) $(BLAS_LDFLAGS)

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

.PHONY: blas-info
blas-info:
	@echo "BLAS_MODE=$(BLAS_MODE)"
	@echo "BLAS_CFLAGS=$(BLAS_CFLAGS)"
	@echo "BLAS_LDFLAGS=$(BLAS_LDFLAGS)"
	@if [ "$(BLAS_MODE)" = "none" ]; then \
		echo "BLAS disabled. Use one of:"; \
		echo "  make build BLAS=accelerate"; \
		echo "  make build BLAS=openblas"; \
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
metal-lib: check-metal-toolchain $(METAL_SRC)
	xcrun -sdk macosx metal -std=metal3.0 -O0 -c $(METAL_SRC) -o $(METAL_AIR)
	xcrun -sdk macosx metallib $(METAL_AIR) -o $(METAL_LIB)

.PHONY: build-metal
build-metal: metal-lib $(METAL_SOURCES)
	$(CC) $(CFLAGS) $(OPENMP_CFLAGS) $(BLAS_CFLAGS) -DGLM_ENABLE_METAL=1 -x objective-c -fobjc-arc -o $(BIN) $(METAL_SOURCES) -framework Foundation -framework Metal $(LDFLAGS) $(OPENMP_LDFLAGS) $(BLAS_LDFLAGS)
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

.PHONY: bench-metal-phases
bench-metal-phases: build-metal
	@set -e; \
	tokens="$(BENCH_TOKENS)"; \
	prompt="$(BENCH_PROMPT)"; \
	cpu_time=/tmp/glm_phase_cpu.time.txt; \
	metal_time=/tmp/glm_phase_metal.time.txt; \
	metal_native_time=/tmp/glm_phase_metal_native.time.txt; \
	env OMP_NUM_THREADS=$(BENCH_OMP_THREADS) OMP_PROC_BIND=$(BENCH_OMP_PROC_BIND) OMP_PLACES=$(BENCH_OMP_PLACES) \
		/usr/bin/time -p ./$(BIN) "$(NATIVE_MODEL)" --backend cpu -n $$tokens -t 0 --seed 0 -i "$$prompt" > /tmp/glm_phase_cpu.txt 2> $$cpu_time; \
	/usr/bin/time -p ./$(BIN) "$(NATIVE_MODEL)" --backend metal -n $$tokens -t 0 --seed 0 -i "$$prompt" > /tmp/glm_phase_metal.txt 2> $$metal_time; \
	if [ "$(METAL_NATIVE)" = "1" ]; then \
		GLM_METAL_NATIVE=1 /usr/bin/time -p ./$(BIN) "$(NATIVE_MODEL)" --backend metal -n $$tokens -t 0 --seed 0 -i "$$prompt" > /tmp/glm_phase_metal_native.txt 2> $$metal_native_time; \
	fi; \
	cpu_tps=$$(awk -v n=$$tokens '/^real /{if ($$2 > 0) printf "%.3f", n/$$2}' $$cpu_time); \
	metal_tps=$$(awk -v n=$$tokens '/^real /{if ($$2 > 0) printf "%.3f", n/$$2}' $$metal_time); \
	echo "phase bench cpu tok/s=$$cpu_tps"; \
	echo "phase bench metal tok/s=$$metal_tps"; \
	if [ "$(METAL_NATIVE)" = "1" ]; then \
		metal_native_tps=$$(awk -v n=$$tokens '/^real /{if ($$2 > 0) printf "%.3f", n/$$2}' $$metal_native_time); \
		echo "phase bench metal(native) tok/s=$$metal_native_tps"; \
	fi; \
	echo "[artifacts] /tmp/glm_phase_cpu.txt $$cpu_time /tmp/glm_phase_metal.txt $$metal_time /tmp/glm_phase_metal_native.txt $$metal_native_time"

.PHONY: bench-tps-cpu
bench-tps-cpu: build
	@set -e; \
	tokens="$(BENCH_TOKENS)"; \
	prompt="$(BENCH_PROMPT)"; \
	out=/tmp/glm_tps_glm-cpu.txt; \
	timef=/tmp/glm_tps_glm-cpu.time.txt; \
	if [ "$(BENCH_WARMUP)" = "1" ]; then \
		env OMP_NUM_THREADS=$(BENCH_OMP_THREADS) OMP_PROC_BIND=$(BENCH_OMP_PROC_BIND) OMP_PLACES=$(BENCH_OMP_PLACES) \
			./glm4.7-flash "$(NATIVE_MODEL)" --backend cpu -n 1 -t 0 --seed 0 -i "$$prompt" > /dev/null; \
	fi; \
	env OMP_NUM_THREADS=$(BENCH_OMP_THREADS) OMP_PROC_BIND=$(BENCH_OMP_PROC_BIND) OMP_PLACES=$(BENCH_OMP_PLACES) \
		/usr/bin/time -p ./glm4.7-flash "$(NATIVE_MODEL)" --backend cpu -n $$tokens -t 0 --seed 0 -i "$$prompt" > $$out 2> $$timef; \
	tps=$$(awk -v n=$$tokens '/^real /{if ($$2 > 0) printf "%.3f", n/$$2}' $$timef); \
	echo "glm.c cpu tok/s=$$tps (omp=$(BENCH_OMP_THREADS), tokens=$$tokens)"; \
	echo "[artifacts] $$out $$timef"

.PHONY: bench-tps-metal
bench-tps-metal: build-metal
	@set -e; \
	tokens="$(BENCH_TOKENS)"; \
	prompt="$(BENCH_PROMPT)"; \
	out=/tmp/glm_tps_glm-metal.txt; \
	timef=/tmp/glm_tps_glm-metal.time.txt; \
	if [ "$(BENCH_WARMUP)" = "1" ]; then \
		./glm4.7-flash "$(NATIVE_MODEL)" --backend metal -n 1 -t 0 --seed 0 -i "$$prompt" > /dev/null; \
	fi; \
	/usr/bin/time -p ./glm4.7-flash "$(NATIVE_MODEL)" --backend metal -n $$tokens -t 0 --seed 0 -i "$$prompt" > $$out 2> $$timef; \
	tps=$$(awk -v n=$$tokens '/^real /{if ($$2 > 0) printf "%.3f", n/$$2}' $$timef); \
	echo "glm.c metal tok/s=$$tps (tokens=$$tokens)"; \
	echo "[artifacts] $$out $$timef"

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

.PHONY: bench-prefill-cpu
bench-prefill-cpu: build
	@set -e; \
	prompt="$(BENCH_PREFILL_PROMPT)"; \
	out=/tmp/glm_prefill_cpu.txt; \
	timef=/tmp/glm_prefill_cpu.time.txt; \
	env OMP_NUM_THREADS=$(BENCH_OMP_THREADS) OMP_PROC_BIND=$(BENCH_OMP_PROC_BIND) OMP_PLACES=$(BENCH_OMP_PLACES) \
		/usr/bin/time -p ./$(BIN) "$(NATIVE_MODEL)" --backend cpu -n 0 -t 0 --seed 0 -i "$$prompt" > $$out 2> $$timef; \
	echo "prefill cpu done"; \
	echo "[artifacts] $$out $$timef"

.PHONY: bench-prefill-metal
bench-prefill-metal: build-metal
	@set -e; \
	prompt="$(BENCH_PREFILL_PROMPT)"; \
	out=/tmp/glm_prefill_metal.txt; \
	timef=/tmp/glm_prefill_metal.time.txt; \
	/usr/bin/time -p ./$(BIN) "$(NATIVE_MODEL)" --backend metal -n 0 -t 0 --seed 0 -i "$$prompt" > $$out 2> $$timef; \
	echo "prefill metal done"; \
	echo "[artifacts] $$out $$timef"

.PHONY: bench-decode-cpu
bench-decode-cpu: build
	@set -e; \
	tokens="$(BENCH_TOKENS)"; \
	prompt="$(BENCH_PROMPT)"; \
	out=/tmp/glm_decode_cpu.txt; \
	timef=/tmp/glm_decode_cpu.time.txt; \
	env OMP_NUM_THREADS=$(BENCH_OMP_THREADS) OMP_PROC_BIND=$(BENCH_OMP_PROC_BIND) OMP_PLACES=$(BENCH_OMP_PLACES) \
		/usr/bin/time -p ./$(BIN) "$(NATIVE_MODEL)" --backend cpu -n $$tokens -t 0 --seed 0 -i "$$prompt" > $$out 2> $$timef; \
	echo "decode cpu done"; \
	echo "[artifacts] $$out $$timef"

.PHONY: bench-decode-metal
bench-decode-metal: build-metal
	@set -e; \
	tokens="$(BENCH_TOKENS)"; \
	prompt="$(BENCH_PROMPT)"; \
	out=/tmp/glm_decode_metal.txt; \
	timef=/tmp/glm_decode_metal.time.txt; \
	/usr/bin/time -p ./$(BIN) "$(NATIVE_MODEL)" --backend metal -n $$tokens -t 0 --seed 0 -i "$$prompt" > $$out 2> $$timef; \
	echo "decode metal done"; \
	echo "[artifacts] $$out $$timef"

.PHONY: bench-kernel-matvec
bench-kernel-matvec: build-metal
	@set -e; \
	tokens="$(BENCH_TOKENS)"; \
	prompt="$(BENCH_PROMPT)"; \
	out=/tmp/glm_matvec_proxy_metal.txt; \
	timef=/tmp/glm_matvec_proxy_metal.time.txt; \
	/usr/bin/time -p ./$(BIN) "$(NATIVE_MODEL)" --backend metal -n $$tokens -t 0 --seed 0 -i "$$prompt" > $$out 2> $$timef; \
	echo "matvec proxy benchmark done (full decode path; use for relative kernel tuning only)"; \
	echo "[artifacts] $$out $$timef"

.PHONY: capture-metal
capture-metal: build-metal
	@set -e; \
	out=/tmp/glm_metal_capture.txt; \
	echo "Running with MTL_CAPTURE_ENABLED=1 ..."; \
	MTL_CAPTURE_ENABLED=1 ./$(BIN) "$(NATIVE_MODEL)" --backend metal -n 8 -t 0 --seed 0 -i "$(CAPTURE_PROMPT)" > $$out; \
	echo "Capture run complete."; \
	echo "Open Xcode GPU capture tools and inspect this run context."; \
	echo "[artifact] $$out"

.PHONY: bench-summary
bench-summary:
	BENCH_TOKENS=$(BENCH_TOKENS) python3 scripts/bench_summary.py
	@echo "[artifacts] /tmp/glm_bench_summary.json /tmp/glm_bench_summary.csv"

.PHONY: clean
clean:
	rm -f $(BIN) $(METAL_AIR) $(METAL_LIB)

# Kernel microbenchmark targets
BENCH_KERNEL_ITERS ?= 128
BENCH_KERNEL_WARMUP ?= 16

.PHONY: bench-micro-matvec
bench-micro-matvec: build-metal
	@./$(BIN) "$(NATIVE_MODEL)" --backend metal --bench-kernel matvec_q80_rows \
		--bench-iters $(BENCH_KERNEL_ITERS) --bench-warmup $(BENCH_KERNEL_WARMUP)

.PHONY: bench-micro-matvec-simdgroup
bench-micro-matvec-simdgroup: build-metal
	@./$(BIN) "$(NATIVE_MODEL)" --backend metal --bench-kernel matvec_q80_rows_simdgroup \
		--bench-iters $(BENCH_KERNEL_ITERS) --bench-warmup $(BENCH_KERNEL_WARMUP)

.PHONY: bench-kernel-rmsnorm
bench-kernel-rmsnorm: build-metal
	@./$(BIN) "$(NATIVE_MODEL)" --backend metal --bench-kernel rmsnorm_f32 \
		--bench-iters $(BENCH_KERNEL_ITERS) --bench-warmup $(BENCH_KERNEL_WARMUP)

.PHONY: bench-kernel-rope
bench-kernel-rope: build-metal
	@./$(BIN) "$(NATIVE_MODEL)" --backend metal --bench-kernel rope_inplace \
		--bench-iters $(BENCH_KERNEL_ITERS) --bench-warmup $(BENCH_KERNEL_WARMUP)

.PHONY: bench-kernel-attention
bench-kernel-attention: build-metal
	@./$(BIN) "$(NATIVE_MODEL)" --backend metal --bench-kernel attention_scores_context \
		--bench-iters 64 --bench-warmup 8

.PHONY: bench-micro-all
bench-micro-all: bench-micro-matvec bench-micro-matvec-simdgroup bench-kernel-rmsnorm bench-kernel-rope bench-kernel-attention

# Compare matvec variants
.PHONY: bench-micro-matvec-compare
bench-micro-matvec-compare: build-metal
	@echo "=== Baseline matvec ==="
	@./$(BIN) "$(NATIVE_MODEL)" --backend metal --bench-kernel matvec_q80_rows \
		--bench-iters $(BENCH_KERNEL_ITERS) --bench-warmup $(BENCH_KERNEL_WARMUP)
	@echo ""
	@echo "=== SIMD-group matvec ==="
	@./$(BIN) "$(NATIVE_MODEL)" --backend metal --bench-kernel matvec_q80_rows_simdgroup \
		--bench-iters $(BENCH_KERNEL_ITERS) --bench-warmup $(BENCH_KERNEL_WARMUP)

# Per-token latency benchmark targets
.PHONY: bench-latency-full
bench-latency-full: build-metal
	@./$(BIN) "$(NATIVE_MODEL)" --backend metal -n $(BENCH_TOKENS) \
		--bench-mode full -i "$(BENCH_PROMPT)" --bench-report /tmp/glm_bench_full.json
	@echo "[artifact] /tmp/glm_bench_full.json"

.PHONY: bench-latency-decode
bench-latency-decode: build-metal
	@./$(BIN) "$(NATIVE_MODEL)" --backend metal -n $(BENCH_TOKENS) \
		--bench-mode decode -i "$(BENCH_PROMPT)" --bench-report /tmp/glm_bench_decode.json
	@echo "[artifact] /tmp/glm_bench_decode.json"

.PHONY: bench-latency-prefill
bench-latency-prefill: build-metal
	@./$(BIN) "$(NATIVE_MODEL)" --backend metal -n 0 \
		--bench-mode prefill -i "$(BENCH_PREFILL_PROMPT)" --bench-report /tmp/glm_bench_prefill.json
	@echo "[artifact] /tmp/glm_bench_prefill.json"
