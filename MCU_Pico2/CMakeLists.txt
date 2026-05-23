# Generated Cmake Pico project file

cmake_minimum_required(VERSION 3.13)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Initialise pico_sdk from installed location
# (note this can come from environment, CMake cache etc)

# == DO NOT EDIT THE FOLLOWING LINES for the Raspberry Pi Pico VS Code Extension to work ==
if(WIN32)
    set(USERHOME $ENV{USERPROFILE})
else()
    set(USERHOME $ENV{HOME})
endif()
set(sdkVersion 2.2.0)
set(toolchainVersion 14_2_Rel1)
set(picotoolVersion 2.2.0-a4)
set(picoVscode ${USERHOME}/.pico-sdk/cmake/pico-vscode.cmake)
if (EXISTS ${picoVscode})
    include(${picoVscode})
endif()
# ====================================================================================
set(PICO_BOARD pico2 CACHE STRING "Board type")

# Pull in Raspberry Pi Pico SDK (must be before project)
include(pico_sdk_import.cmake)

project(drone_Pico2 C CXX ASM)

# Initialise the Raspberry Pi Pico SDK
pico_sdk_init()

# TEST_MODE uses explicit numeric values only:
# 0 = field mode (UART0 GP12/GP13), 1 = test mode (USB stdio)
set(TEST_MODE "0" CACHE STRING "0=field(UART0 GP12/GP13), 1=test(USB stdio)")
set_property(CACHE TEST_MODE PROPERTY STRINGS 0 1)

# Add executable. Default name is the project name, version 0.1

add_executable(drone_Pico2 drone_Pico2.c )

pico_set_program_name(drone_Pico2 "drone_Pico2")
pico_set_program_version(drone_Pico2 "0.1")

# Modify the below lines to enable/disable output over UART/USB
if(TEST_MODE STREQUAL "1")
    pico_enable_stdio_uart(drone_Pico2 0)
    pico_enable_stdio_usb(drone_Pico2 1)
else()
    pico_enable_stdio_uart(drone_Pico2 1)
    # Keep USB enabled in field mode so picotool "Run Project" can still reboot/flash.
    pico_enable_stdio_usb(drone_Pico2 1)
    target_compile_definitions(drone_Pico2 PRIVATE
            PICO_DEFAULT_UART=0
            PICO_DEFAULT_UART_TX_PIN=12
            PICO_DEFAULT_UART_RX_PIN=13
            PICO_DEFAULT_UART_BAUD_RATE=57600)
endif()

# Define TEST_MODE macro from numeric cache value directly.
target_compile_definitions(drone_Pico2 PRIVATE TEST_MODE=${TEST_MODE})

# Add the standard library to the build
target_link_libraries(drone_Pico2
        pico_stdlib
        hardware_pwm
        hardware_clocks)

# Add the standard include files to the build
target_include_directories(drone_Pico2 PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}
)

pico_add_extra_outputs(drone_Pico2)

