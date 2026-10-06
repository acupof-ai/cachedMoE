# Resolve legacy cache keys before a canonical option supplies its default.
# A legacy-only value stays a normal variable: creating a canonical cache entry
# here would prevent a later -DDEEPMOE_<name> change in this same build tree.
# Once a canonical normal or cache variable exists, it wins, including "".
# This also means a canonical cached default wins over a later legacy -D.
include_guard(GLOBAL)

function(cachedmoe_resolve_cache suffix type description)
    set(canonical "CACHEDMOE_${suffix}")
    set(legacy "DEEPMOE_${suffix}")
    if(DEFINED CACHE{${legacy}})
        # Normal expansion marks the CLI key consumed for CMake's unused-key
        # check. Resolve from CACHE separately so an old normal variable cannot
        # shadow the legacy cache value selected by this compatibility layer.
        set(legacy_cli_use "${${legacy}}")
        set(legacy_value "$CACHE{${legacy}}")
    endif()

    if(DEFINED ${canonical})
        # Like option()/set(... CACHE ...), give an untyped command-line cache
        # entry its declared type. Never FORCE over an existing typed value.
        if(DEFINED CACHE{${canonical}})
            get_property(cache_type CACHE "${canonical}" PROPERTY TYPE)
            if(cache_type STREQUAL "UNINITIALIZED")
                set(cache_value "$CACHE{${canonical}}")
                set("${canonical}" "${cache_value}" CACHE "${type}" "${description}")
            endif()
        endif()
        set(reason "ignored because ${canonical} is defined")
    elseif(DEFINED CACHE{${legacy}})
        get_property(cache_type CACHE "${legacy}" PROPERTY TYPE)
        set(cache_value "${legacy_value}")
        if(cache_type STREQUAL "UNINITIALIZED")
            # FILEPATH/PATH conversion follows CMake's own untyped -D rules.
            # Typed legacy entries retain both their original type and value.
            set("${legacy}" "${cache_value}" CACHE "${type}" "${description}")
            set(cache_value "$CACHE{${legacy}}")
        endif()
        set("${canonical}" "${cache_value}" PARENT_SCOPE)
        set(reason "used as the fallback for ${canonical}")
    endif()

    if(DEFINED CACHE{${legacy}})
        # A shader helper can resolve the same key in several function scopes.
        # Warn once per configure, without printing user-provided cache values.
        get_property(warned GLOBAL PROPERTY "CACHEDMOE_CACHE_WARNED_${suffix}")
        if(NOT warned)
            message(WARNING "${legacy} is deprecated; use ${canonical}. It is ${reason}.")
            set_property(GLOBAL PROPERTY "CACHEDMOE_CACHE_WARNED_${suffix}" TRUE)
        endif()
    endif()
endfunction()

macro(cachedmoe_option suffix description default)
    cachedmoe_resolve_cache("${suffix}" BOOL "${description}")
    # CMP0077 is NEW under the project's CMake 3.25 minimum: option() respects
    # the normal variable supplied by a legacy-only cache entry.
    option(CACHEDMOE_${suffix} "${description}" "${default}")
endmacro()

macro(cachedmoe_cache suffix type description default)
    cachedmoe_resolve_cache("${suffix}" "${type}" "${description}")
    if(NOT DEFINED CACHEDMOE_${suffix})
        set(CACHEDMOE_${suffix} "${default}" CACHE "${type}" "${description}")
    endif()
endmacro()
