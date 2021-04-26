#getwd()
filename = "2019_Response_to_Resistance_Combined.csv"
force_data = read.csv(filename, stringsAsFactors = T)
str(force_data)

# Use Cameron's Time from preprocessing
sum(is.na(force_data$Time..Occurred)) #1434
sum(is.na(force_data$Time..Occurred))/nrow(force_data) #0.2397592

force_data$Time..Occurred <- as.character(force_data$Time..Occurred)
### Pad out 0's in the beginning
for(time in 1:length(force_data$Time..Occurred)){
  if(!is.na(force_data$Time..Occurred[time])){
    while(nchar(force_data$Time..Occurred[time]) < 4){
      force_data$Time..Occurred[time] <- paste0(rep("0",nchar(force_data$Time..Occurred[time])),force_data$Time..Occurred[time])
    }
  }
}
force_data$Time_Occ <- sub("(\\d+)(\\d{2})", "\\1:\\2", force_data$Time..Occurred)

#Use definition of day and night from Cameron's Aim 1: 
force_data$parsed_time = as.numeric(gsub(":.*$","",force_data$Time_Occ))
force_data$Time_of_day <- "Night"
force_data$Time_of_day[which(force_data$parsed_time > 6 & force_data$parsed_time < 21)] <- "Day"
force_data$Time_of_day = as.factor(force_data$Time_of_day)
table(force_data$Time_of_day) # Day: 2067, Night 3914

#Fix Race to single variable like other datasets 
#table(force_data$Subject..Race)
#table(force_data$Subject..Ethnicity)
force_data$Race <- as.character(force_data$Subject..Race)
force_data$Race[which((force_data$Race == "W" | force_data$Race == "U") & force_data$Subject..Ethnicity == "H")] <- "H"
#table(force_data$Race)

#Rename race categories 
force_data$Race[which(force_data$Race == "W")] <- "WHITE"
force_data$Race[which(force_data$Race == "U")] <- "UNKNOWN"
force_data$Race[which(force_data$Race == "")]  <- "UNKNOWN"
force_data$Race[which(force_data$Race == "B")] <- "BLACK"
force_data$Race[which(force_data$Race == "A")] <- "ASIAN"
force_data$Race[which(force_data$Race == "H")] <- "HISPANIC OR LATINO"
force_data$Race[which(force_data$Race == "M")] <- "MIDDLE EASTERN"
force_data$Race[which(force_data$Race == "P")] <- "HAWAIIAN/PACIFIC ISLANDER" 
force_data$Race[which(force_data$Race == "I")] <- "AMERICAN INDIAN/ALASKAN NATIVE"
force_data$Race = as.factor(force_data$Race)
#table(force_data$Race)

filter = "Nature of Contact"
predictors = "Time_of_day, Area_Command, ZIP, Officer Yrs of Service, Officer  Organization Desc, Subject  Sex, Race, Subject Conduct Desc, Subject Resistance, Nature.of.Contact, Reason.Desc"
targets = "R2R Level, Subject Effects, Weapon Used 1" #will have to consolidate weapons variables... 

# fix force level to match requirements of xgboost
library(plyr)
force_data$R2R.Level = as.character(force_data$R2R.Level)
force_data$R2R.Level = revalue(force_data$R2R.Level, c("23"="2", "24"="2", "34"="3", "234"="2"))
force_data$R2R.Level = as.integer(force_data$R2R.Level)-1
#table(force_data$flevel)

#create a new binary feature for high levels of force 
force_data$high_force = 0 
force_data$high_force[which(force_data$R2R.Level <= 1)] = 1 #force levels have been adjusted to 0,1,2,3. 

#create separate dataset for only traffic stops 
traffic_data = force_data[which(force_data$Nature.of.Contact == "1-TRAFFIC STOP [Motor Vehicle Stop]"),]

## create stratified samples 
library(dplyr)
#strat sampling for traffic data 
traffic_data.x.strat  <- traffic_data %>% group_by(traffic_data$high_force) %>% sample_n(size=sum(traffic_data$high_force==1))
traffic_data.y.strat  <- traffic_data.x.strat$high_force
table(traffic_data.y.strat) #20 and 20 
#strat sampling for total dataset 
force_data.x.strat  <- force_data %>% group_by(force_data$high_force) %>% sample_n(size=sum(force_data$high_force==1))
force_data.y.strat  <- force_data.x.strat$high_force
table(force_data.y.strat) #208 and 208 

# Using xgboost
library(xgboost)
library(Matrix)

#Full model 
#subset(force_data, select=c(R2R.Level, Time_of_day, Area_Command, ZIP, Officer.Yrs.of.Service, Officer..Organization.Desc, Subject..Sex, Race, Subject.Conduct.Desc, Subject.Resistance, Nature.of.Contact, Reason.Desc)) #most important predictor was years of service. Loss was ~0.2
#
#subset(force_data, select=c(R2R.Level, Time_of_day, Area_Command, ZIP, Officer..Organization.Desc, Subject..Sex, Race, Subject.Conduct.Desc, Subject.Resistance, Nature.of.Contact, Reason.Desc)) # loss was 0.288. Reason for force:in custody, maintaining control. Second is female. Third is "Defensive resistance" for subject resistance type. Race is the 7th most important predictor... 
#
#subset(force_data, select=c(high_force, Time_of_day, Area_Command, ZIP, Officer..Organization.Desc, Subject..Sex, Race, Subject.Conduct.Desc, Subject.Resistance, Nature.of.Contact, Reason.Desc)) #ONLY LOOKING AT FORCE <2. Loss was 0.072887. Defensive resistance, alcohol, maintaining control, dispatched call, Male, Night time. Then seven is White. 
#


#select the subset of features for the model
df = subset(traffic_data.x.strat, select=c(high_force, Time_of_day, ZIP, Subject..Sex, Race, Subject.Conduct.Desc, Subject.Resistance,  Reason.Desc))
#convert subset of features to a numerical matrix
boost.x.data = sparse.model.matrix(high_force ~ ., data = df)[,-1]
boost.y.data = as.matrix(subset(traffic_data.x.strat, select=c(high_force)))

niter = 2000 
set.seed(99)
boost.train <- xgboost(data = boost.x.data, 
                                 label = boost.y.data,
                                 max.depth = 10,
                                 nrounds = niter,
                                 objective = "reg:logistic", #The other function: multi:softmax
                                 #eval_metric = 'mlogloss',
                                 eta = 0.1, 
                                 verbose = 0,
                                 num_class = 1) #this is the number of states your output variable can take.

boost.train.loss <- boost.train$evaluation_log$train_rmse[niter]
importance_matrix = xgb.importance(model=boost.train)
print(importance_matrix[1:10,])
xgb.plot.importance(importance_matrix = importance_matrix[1:10,])


#When you look at unstratified, all force including non-motor vehicles, most force is just get drunk males at night using defensive resistane. Basically no mention of race. 
#Looking at stratified sample, and only motor vehicles, the importance is: alcohol, defensive resistance, vehicle in pursuit, black, latino, female. This is a 40 person sample, and race is only the 4th most important predictor, and this is observational data so race is probably a stand-in for something else or a random fluctuation... 

## Next up: predict Subject.Effects 
#traffic_data.x.strat
#force_data.x.strat

#encode Subject.Effects as numerical for xgboost 
table(force_data$Subject.Effects)
force_data$Subject.Effects = as.character(force_data$Subject.Effects)
force_data$SubjectInjury = revalue(force_data$Subject.Effects, c("NO COMPLAINT OF INJURY/PAIN"=0,
                                                                 "COMPLAINT OF INJURY/PAIN BUT NONE OBSERVED"=1,
                                                                 "COMPLAINT OF INJURY/PAIN"=2, 
                                                                 "MINOR INJURY"=3,
                                                                 "SERIOUS INJURY"=4,
                                                                 "DEATH"=5,
                                                                 '9'=""))
force_data$SubjectInjury[which(force_data$SubjectInjury=="")] = NA
table(force_data$SubjectInjury)


df = subset(force_data, select=c(SubjectInjury, R2R.Level, Time_of_day, ZIP, Subject..Sex, Race, Subject.Conduct.Desc, Subject.Resistance,  Reason.Desc))
boost.x.data = sparse.model.matrix(SubjectInjury ~ ., data = df)[,-1]
boost.y.data = as.matrix(na.omit(subset(df, select=c(SubjectInjury))))

niter = 1000 
set.seed(99)
boost.train <- xgboost(data = boost.x.data, 
                       label = boost.y.data,
                       max.depth = 10,
                       nrounds = niter,
                       objective = "multi:softmax",  #The other function: "reg:logistic"
                       eval_metric = 'mlogloss',
                       eta = 0.1, 
                       verbose = 0,
                       num_class = 6) #this is the number of states your output variable can take.

boost.train.loss <- boost.train$evaluation_log$train_mlogloss[niter]
importance_matrix = xgb.importance(model=boost.train)
print(importance_matrix[1:10,])
xgb.plot.importance(importance_matrix = importance_matrix[1:10,])






