---
title: "Cran Comments"
author: "Ivan Svetunkov"
date: "20 June 2026"
output: html_document
---

## Version
This is ``smooth`` package, v4.5.0

This release retires the legacy occurrence-ETS code path.


## Test environments
* local Ubuntu 26.04, R 4.6.0
* github actions
* win-builder (devel and release)
* rhub v2

## R CMD check results
R CMD check results
checking installed package size ... NOTE
    installed size is 20.8Mb
    sub-directories of 1Mb or more:
      R      1.2Mb
      doc    3.3Mb
      libs  15.7Mb
0 errors | 0 warnings | 1 note

## Github actions
Successful checks for:

- Windows latest release with latest R
- MacOS 15.7.3 with latest R
- Ubuntu 24.04.4 LTS with latest R

## win-builder check results
>* checking package dependencies ... NOTE
>Package suggested but not available for checking: 'doMC'

This is expected, because doMC is not available for Windows.

## R-hub
Successful checks for:

- Windows Server 2022 x64 (build 26100), R 4.5.0
- MacOS macOS Sequoia 15.7.7, R 4.5.0
- MacOS 15.7.3, R 4.5.0
- Ubuntu 24.04.4 LTS, R 4.5.0

## Downstream dependencies
I have also run R CMD check on reverse dependencies of smooth.
No ERRORs or WARNINGs found.
