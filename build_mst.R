library(stats)

getMST <- function(input_distance_df){
  input_distance_matrix <- as.matrix(input_distance_df)
  distMat <- as.dist(input_distance_matrix, diag = FALSE, upper = FALSE)
  M <- ape::mst(distMat)
  M[lower.tri(M)] <- 0L
  net <- which(M == 1L, arr.ind=TRUE)
  rownames(net) <- NULL
  #runs <- attr(distMat, "Labels")
  runs <- colnames(M)
  net <- cbind("A" = runs[net[,1]], "B" = runs[net[,2]])
  net
}