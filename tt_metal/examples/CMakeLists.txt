install(
    DIRECTORY
        hello_world_compute_kernel
    # DESTINATION ${CMAKE_INSTALL_DOCDIR}/examples
    # FIXME(afuller): Something funky is happening when installing files into /usr/share/doc on a default Docker image.
    #                 Speculation: some dependency for magic doc handling is missing.
    #                 For now keep it out of 'doc'.
    DESTINATION /usr/share/tt-metalium/examples
    COMPONENT metalium-examples
)
