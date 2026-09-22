# Build the SIPNET binary that pySIPNET drives.
#
# SIPNET lives in the sipnet/ git submodule, pinned to a specific release tag.
# From SIPNET v2.0.0 onward every model option (snow, litter pool, nitrogen
# cycle, and so on) is chosen at run time in the sipnet.in config file, so
# there is exactly one binary to build and no compiler flags to pass. Earlier
# SIPNET versions chose those options at compile time, which is why older
# versions of this Makefile built one binary per option combination.

SIPNET_DIR := sipnet
CACHE_DIR  := .sipnet_cache
# The binary lives in a subdirectory named by the pinned commit, read from
# pysipnet/version.py so it is the same name pysipnet.build looks under. A pin
# bump therefore lands the new binary in an empty directory instead of
# overwriting, or being mistaken for, the old one.
COMMIT     := $(shell sed -n 's/^SIPNET_PINNED_COMMIT: str = "\([0-9a-f]\{12\}\).*/\1/p' pysipnet/version.py)
ifeq ($(COMMIT),)
$(error could not read SIPNET_PINNED_COMMIT from pysipnet/version.py)
endif
BINARY     := $(CACHE_DIR)/$(COMMIT)/sipnet

.PHONY: sipnet sipnet-download submodule clean-sipnet

# Default target: build SIPNET and copy the binary into the cache directory
# where pysipnet.runner looks for it.
sipnet: submodule
	$(MAKE) -C $(SIPNET_DIR) clean
	$(MAKE) -C $(SIPNET_DIR)
	mkdir -p $(CACHE_DIR)/$(COMMIT)
	# Install via a temporary name and mv, never by copying over the target.
	# On Apple Silicon every Mach-O binary carries a code signature, and
	# overwriting one in place invalidates it — the kernel then SIGKILLs the
	# binary on exec, with no error message to explain why. mv replaces the
	# directory entry instead, so the signature stays intact. This also means a
	# failed build cannot leave a half-written binary behind.
	cp $(SIPNET_DIR)/sipnet $(BINARY).incoming
	mv $(BINARY).incoming $(BINARY)
	@echo "Built: $(BINARY)"
	@$(BINARY) --version

# Fetch a prebuilt binary instead of compiling. Useful on a machine without a
# C toolchain. The archive's SHA-256 is pinned in pysipnet/version.py and
# checked before anything is unpacked; compiling from source (above) works on
# any platform and is the default.
sipnet-download:
	uv run pysipnet install-sipnet --method download --force

# Fetch the submodule contents if this is a fresh clone.
submodule:
	@test -f $(SIPNET_DIR)/Makefile || git submodule update --init $(SIPNET_DIR)

clean-sipnet:
	$(MAKE) -C $(SIPNET_DIR) clean
	rm -rf $(CACHE_DIR)
