require(reticulate)
if (!reticulate::py_available(TRUE))
  stop("Unable to call python. To run this package, you must have python on your system (you don't need to know how to program Python, you just need it installed). You can easily download Python version 3 on the official Python website.")

npEstimateBFS <- function(...) {
  sys <- import("sys")
  sys$path <- c(paste0(find.package("npcompare"), "/python"), sys$path)
  npc <- import("npcompare")
  pythonObj <- npc$EstimateBFS(...)
  structure(list(pythonObj=pythonObj), class = c("npEstimateBFS", "npGeneric"))
}

npCompare <- function(...) {
  sys <- import("sys")
  sys$path <- c(paste0(find.package("npcompare"), "/python"), sys$path)
  npc <- import("npcompare")
  pythonObj <- npc$Compare(...)
  #estimate$sample <- function(...) {

  #}
  structure(list(pythonObj=pythonObj), class = c("npCompare", "npGeneric"))
}

npSave <- function(obj, file = ".RData") {
  print("Attention: only load this object using npLoad(obj).")
  #pickle dump to tmp file
  tmp_file <- tempfile()
  f <- import_builtins()$open(tmp_file, "wb")
  pickle <- import("pickle")
  pickle$dump(obj$pythonObj, f)
  f$close()

  #open pickle dump as binary and save it with S3 class information
  fsize <- file.size(tmp_file)
  fconn <- file(tmp_file, "rb")
  frdta <- list(readBin(fconn, "raw", fsize), obj)
  close(fconn)
  save(frdta, file=file)
}

npLoad <- function(file) {
  sys <- import("sys")
  sys$path <- c(paste0(find.package("npcompare"), "/python"), sys$path)
  npc <- import("npcompare")

  load(file)

  #save pickle dump to temporary file
  tmp_file <- tempfile()
  fconn <- file(tmp_file, "wb")
  frdta <- writeBin(frdta[[1]], fconn)
  close(fconn)

  #now load pickle temporary file with pickle
  f <- import_builtins()$open(tmp_file, "rb")
  pickle <- import("pickle")
  recovered_pythonObj <- pickle$load(f)
  f$close()
  frdta$pythonObj <- recovered_pythonObj

  frdta
}
