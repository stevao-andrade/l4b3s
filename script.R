# lendo os dados de treinamento
train <- read.csv("train.csv")
test <- read.csv("test.csv")


# deixando o conjunto de treinamento com 5%, para nao ter gargalo de processamento
trainRowCount <- floor(0.05 * nrow(train))
set.seed(1)
trainIndex <- sample(1:nrow(train), trainRowCount)
train <- train[trainIndex,]

# criando conjunto de treinamento e teste, para executar sem necessidade do kaggle
trainRowCount <- floor(0.7 * nrow(train))
set.seed(1)
trainIndex <- sample(1:nrow(train), trainRowCount)
train_70perc <- train[trainIndex,]
test_30perc <- train[-trainIndex,]
test_30perc <- test_30perc[, -2]   # removendo a coluna target de teste

# substituindo valores NAs
#  - numericos: substituiu pela media
#  - simbolicos: subistitiu pela moda
# usando funcao do pacote randomForest
train_70perc <- na.roughfix(train_70perc)
test_30perc <- na.roughfix(test_30perc)

# Usando apenas colunas numericas
train_70perc <- train_70perc[, sapply(train_70perc, is.numeric)]
test_30perc <- test_30perc[, sapply(test_30perc, is.numeric)]

#removendo colunas simbolicas com numero de valores > 56 (req do algoritmo)
# e com numero de valores diferentes no conjunto de teste e treinamento
remove1 <- c("v22", "v56", "v125", "v79", "v47")

# removendo colunas com correlacao maior que 95%
remove2 <- c("v46", "v63", "v53", "v64", "v76", "v54", "v89", "v105", "v60", "v96",
            "v83", "v114", "v116", "v63", "v64", "v106", "v89", "v100", "v76",
            "v115", "v121", "v95", "v118", "v128")

# remvendo coluna simbolica duplicada
remove3 <- c("v107")

#correlacao > 98%
remove4 <- c("v46", "v11", "v76")

# removendo o conjunto desejado
train_70perc <- train_70perc[, -which(names(train_70perc) %in% remove4)]
test_30perc <- test_30perc[, -which(names(test_30perc) %in% remove4)]

# construcao do modelo
rf <- randomForest(as.factor(target) ~ ., data = train_70perc, ntree = 10)
yhat <- predict(rf, test_30perc, type="prob")[,2]

# imprimindo matriz de confusao
print(rf)

# gerando arquivo de saida
write.csv(data.frame(ID = test_30perc$ID, PredictedProb = yhat), "submission_rf.csv", row.names = F)


# removendo as colunas definidas no preprocessamento
train <- train[, -which(names(train) %in% remove1)]
test <- test[, -which(names(test) %in% remove1)]
train <- train[, -which(names(train) %in% remove2)]
test <- test[, -which(names(test) %in% remove2)]
train <- train[, -which(names(train) %in% remove3)]
test <- test[, -which(names(test) %in% remove3)]
train <- train[, -which(names(train) %in% remove4)]
test <- test[, -which(names(test) %in% remove4)]


library(randomForest)

#execução com os arquivos originais
train <- na.roughfix(train)
test <- na.roughfix(test)

train_target0 <- na.roughfix(train_target0)
train_target1 <- na.roughfix(train_target1)

train <- rbind(train_target0, train_target1)


# omitir NAs
train <- na.omit(train);
test <- na.omit(test);



rf <- randomForest(as.factor(target) ~ ., data = train, ntree = 100)
yhat <- predict(rf, test, type="prob")[,2]

print(rf)
yhat

# gerando arquivo de saida
write.csv(data.frame(ID = test$ID, PredictedProb = yhat), "submission_rf_full_v8.csv", row.names = F)


# quantidade de linhas e colunas nos dados
dim(train)

# estrutura dos dados
str(train)

# Distribuicao da variavel target (1: aprovacao rapida, 0: requer analise)
# em valores absolutos:
table(train$target)
# frequencia:
prop.table(table(train$target), margin = NULL)

# dados apenas com target 0
train_target0 <- train[which(train$target == 0),]

# dados apenas com target 0
train_target1 <- train[which(train$target == 1),]

# vetor com as colunas de atributos simbolicos
symbolic.var.names <- colnames(train)[sapply(train, is.factor)]

#vetor com as colunas de atributos numericos
numeric.var.names <- colnames(train)[sapply(train, is.numeric)]

# conjunto de dados com as variaveis categoricas
train.symbolic <- train[, symbolic.var.names]

# conjunto de dados com as variaveis continuas
train.numeric <- train[, numeric.var.names]

# numero de NAs nos atributos
sapply(train, function(x) sum(is.na(x)))
sapply(train.numeric, function(x) sum(is.na(x)))

# porcentagem de NAs
sapply(train, function(x) sum(is.na(x)))/dim(train)[1]
sapply(train.numeric, function(x) sum(is.na(x)))/dim(train.numeric)[1]

# numero de valores diferentes para cada atributo
sapply(train, function(x) length(unique(x)))
sapply(train.numeric, function(x) length(unique(x)))
sapply(train.symbolic, function(x) length(unique(x)))


# quick view nos dados
summary(train);
summary(train.symbolic);
summary(train.numeric);

# correlacoes nos dados numericos
train.numeric.corr = cor(train.numeric, use='complete.obs');

# instalacao do pacote para geracao do grafico de correlacoes
install.packages("corrplot")

#geracao do gráfico
corrplot(train.cat.corr)

table(train$v66) # conta numero de ocorrencias dos atributos simbolicos
table(train$v66, useNA = "always") # conta numero de ocorrencias, incluindo NAs


# bloco para criação de matriz com dados de correlacao maior de 80%
train.matrix.high_corr <- matrix(nrow=200, ncol=3)
i = 1;
j = 1;
k = 1
while  (i < nrow(train.numeric.corr)) {
  j = i+1;
  while(j < ncol(train.numeric.corr)) {
    if (train.numeric.corr[i,j] > 0.95 || train.numeric.corr[i, j] < -0.95) {
      train.matrix.high_corr[k,] <- 
        c(rownames(train.numeric.corr)[i], 
          colnames(train.numeric.corr)[j], 
          train.numeric.corr[i, j]);
      k = k + 1;
    }
    j = j + 1;
  }
  i = i + 1;
}
rm(i)
rm(j)
rm(k)

# relacao entre as variaveis nominais
table(train$v91, train$v107)