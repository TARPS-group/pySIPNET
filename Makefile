SIPNET_DIR     := sipnet
PATCH_SCRIPT   := patches/apply_flags_patch.py
CACHE_DIR      := .sipnet_cache

# Flags always set: HEADER=1 forces a column-header row in output,
# making output parsing unambiguous regardless of parameter changes.
BASE_FLAGS := -DHEADER=1

# Named presets — these map to ModelPreset enum values in pysipnet/runner.py.
# To add a new preset, extend both this Makefile and pysipnet/build.py.
STANDARD_FLAGS := $(BASE_FLAGS) -DSNOW=1 -DGDD=1 -DWATER_HRESP=1 -DGROWTH_RESP=0 -DLITTER_POOL=0 -DLEAF_WATER=0
FOREST_FLAGS   := $(BASE_FLAGS) -DSNOW=1 -DGDD=1 -DWATER_HRESP=1 -DGROWTH_RESP=0 -DLITTER_POOL=1 -DLEAF_WATER=0

.PHONY: sipnet sipnet-standard sipnet-forest patch-sipnet clean-sipnet

sipnet: sipnet-standard sipnet-forest
	@echo "SIPNET binaries built in $(CACHE_DIR)/"

# Apply the #ifndef patch to SIPNET source (idempotent).
patch-sipnet:
	python3 $(PATCH_SCRIPT) $(SIPNET_DIR)

sipnet-standard: patch-sipnet
	$(MAKE) -C $(SIPNET_DIR) clean
	$(MAKE) -C $(SIPNET_DIR) CFLAGS="-Wall -g -Isrc -Wno-c2x-extensions $(STANDARD_FLAGS)"
	mkdir -p $(CACHE_DIR)
	cp $(SIPNET_DIR)/sipnet $(CACHE_DIR)/sipnet_standard
	@echo "Built: $(CACHE_DIR)/sipnet_standard"

sipnet-forest: patch-sipnet
	$(MAKE) -C $(SIPNET_DIR) clean
	$(MAKE) -C $(SIPNET_DIR) CFLAGS="-Wall -g -Isrc -Wno-c2x-extensions $(FOREST_FLAGS)"
	mkdir -p $(CACHE_DIR)
	cp $(SIPNET_DIR)/sipnet $(CACHE_DIR)/sipnet_forest
	@echo "Built: $(CACHE_DIR)/sipnet_forest"

clean-sipnet:
	$(MAKE) -C $(SIPNET_DIR) clean
	rm -rf $(CACHE_DIR)
