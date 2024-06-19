library(ggplot2)

# read data
loss_data <- read.csv("loss.csv")

ggplot(loss_data, aes(x=Epoch)) +
  geom_line(aes(x=Epoch, y=Train_Loss, color="blue"), size=1) +
  geom_line(aes(x=Epoch, y=Test_Loss, color="red"), size=1) +
  theme_minimal() +
  ggtitle("Training and Test Loss During Training") +
  xlab("Epoch") +
  ylab("Loss") +
  scale_color_identity(guide = "legend", labels = c("Train Loss", "Test Loss"))
