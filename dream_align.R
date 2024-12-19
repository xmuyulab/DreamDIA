library(DIAlignR)

# aa <- c(1,2,3,4,5)
# bb <- c(2,3,4,5,6)
# cc <- c(2,4,5,7,8)
# dd <- c(1,3,5,7,8)
# kk <- data.frame(aa, bb, cc, dd)
# colnames(kk) <- c("time", "aa", "bb", "cc")
# my_xics_r <- xics_from_py_to_r(kk)
# my_xics_r
xics_from_py_to_r <- function(xics_from_py) {
  xics_r <- list()
  for (i in 2:dim(xics_from_py)[2]) {
    single_trace <- cbind(xics_from_py$time, xics_from_py[, i])
    colnames(single_trace) <- c("time", colnames(xics_from_py)[i])
    xics_r[[i - 1]] <- single_trace
  }
  xics_r
}

# aa <- c(1,2,3,4,5)
# bb <- c(2,3,4,5,6)
# kk <- data.frame(aa, bb)
# colnames(kk) <- c("x", "y")
# lowess_coef_py_to_r(kk)
lowess_coef_py_to_r <- function(lowess_coef_from_py) {
  lowess_coef_r <- list()
  lowess_coef_r$x <- lowess_coef_from_py$x
  lowess_coef_r$y <- lowess_coef_from_py$y
  lowess_coef_r
}

dream_align_linear <- function(xic.ref, xic.exp, global_fit_arguments) {
  xic.ref.r <- xics_from_py_to_r(xic.ref)
  xic.exp.r <- xics_from_py_to_r(xic.exp)
  params <- paramsDIAlignR()
  params[["globalAlignment"]] <- "linear"
  alignres <- getAlignedTimesFast(xic.ref.r, 
                                  xic.exp.r, 
                                  c(global_fit_arguments[2], global_fit_arguments[1]), 
                                  global_fit_arguments[3], 
                                  params)
  alignres
}

dream_align_lowess <- function(xic.ref, xic.exp, lowess_coef_from_py, adaptive_rt) {
  xic.ref.r <- xics_from_py_to_r(xic.ref)
  xic.exp.r <- xics_from_py_to_r(xic.exp)

  lowess_coef_r <- lowess_coef_py_to_r(lowess_coef_from_py)
  lowess_function <- stats::approxfun(lowess_coef_r, ties = mean)

  params <- paramsDIAlignR()
  params[["globalAlignment"]] <- "loess"

  alignres <- getAlignedTimesFast(xic.ref.r, 
                                  xic.exp.r, 
                                  lowess_function, 
                                  adaptive_rt, 
                                  params)
  alignres
}