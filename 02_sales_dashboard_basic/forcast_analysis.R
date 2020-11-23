# 1.0 LIBRARIES ----

# Core
library(tidyverse)
library(tidyquant)

# Interactive Visualizations
library(plotly)

# Modeling Libraries
library(parsnip)
library(timetk)

# Database
library(odbc)
library(RSQLite)


# 2.0 PROCESSED DATA ----
con <- dbConnect(RSQLite::SQLite(), "Sales_Dashboard/00_data/bikes_database.db")


bikes_tbl      <- tbl(con, "bikes")
bikeshops_tbl  <- tbl (con, "bikeshops")
orderlines_tbl <-  tbl(con, "orderlines")

processed_data_tbl <- orderlines_tbl %>% 
  left_join(bikeshops_tbl, by = c("customer.id" = "bikeshop.id")) %>% 
  left_join(bikes_tbl, by     = c("product.id" = "bike.id")) %>% 
  mutate(extended_price       = quantity * price) %>% 
  collect()

processed_data_tbl <- processed_data_tbl %>% 
  mutate(order.date  = ymd(order.date)) %>% 
  separate(location, 
           into = c("city", "state"), 
           sep  = ", ") %>% 
  separate(description, 
           into = c("category_1", "category_2", "frame_material"), 
           sep  = " - ") %>% 
  select(order.date, order.id, order.line, state, quantity, price,
         extended_price, category_1:frame_material, bikeshop.name)

processed_data_tbl


dbDisconnect(con)


# 3.0 TIME SERIES AGGREGATION ----

# 3.1 DATA MANIPULATION ----
time_unit <- "quarter"

time_plot_tbl <- processed_data_tbl %>% 
  mutate(date = floor_date(order.date, unit = time_unit)) %>% 
  
  group_by(date) %>% 
  summarize(total_sales = sum(extended_price)) %>% 
  ungroup() %>% 
  
  mutate(label_text = str_glue("Date: {date}
                               Revenue: {scales::dollar(total_sales)} "))

time_plot_tbl

# 3.2 FUNCTION ----

aggregate_time_series <- function(data, time_unit = "month") {
  
  output_tbl <- data %>% 
    mutate(date = floor_date(order.date, unit = time_unit)) %>% 
    
    group_by(date) %>% 
    summarize(total_sales = sum(extended_price)) %>% 
    ungroup() %>% 
    
    mutate(label_text = str_glue("Date: {date}
                               Revenue: {scales::dollar(total_sales)} "))

  return(output_tbl)
  }

# 
# processed_data_tbl %>% 
#   aggregate_time_series("quarter")


processed_data_tbl %>% 
  aggregate_time_series(time_unit = "quarter")

# 3.3 TIME SERIES PLOT ----

data <- processed_data_tbl %>% 
  aggregate_time_series("month")

g <- data %>% 
  ggplot(aes(date, total_sales)) + 
  geom_line(color = "#2c3e50") +
  geom_point(aes(text = label_text), color = "#2c3e50", size = 0.1) +
  geom_smooth(method = "loess", span = 0.2) + 
  theme_tq() +
  expand_limits(y = 0) +
  scale_y_continuous(labels = scales::dollar_format()) + 
  labs(x = "", y = "")

ggplotly(g, tooltip = "text")


# 3.4 FUNCTION ----

plot_time_series <- function(data) {
  
  g <- data %>% 
    ggplot(aes(date, total_sales)) + 
    geom_line(color = "#2c3e50") +
    geom_point(aes(text = label_text), color = "#2c3e50", size = 0.1) +
    geom_smooth(method = "loess", span = 0.2) + 
    theme_tq() +
    expand_limits(y = 0) +
    scale_y_continuous(labels = scales::dollar_format()) + 
    labs(x = "", y = "")
  
  ggplotly(g, tooltip = "text")
  
}

processed_data_tbl %>% 
  aggregate_time_series("month") %>% 
  plot_time_series()


# 4.0 FORECAST ----

# 4.1 SETUP TRAINING AND FUTURE DATA ----

# timetk

data <- processed_data_tbl %>% 
  aggregate_time_series(time_unit = "year")

data %>% tk_index() %>% tk_get_timeseries_signature()

data %>% tk_index() %>% tk_get_timeseries_summary() 

tk_get_timeseries_unit_frequency()

data %>% tk_get_timeseries_variables()

data %>% tk_augment_timeseries_signature()

train_tbl <- data %>% 
  tk_augment_timeseries_signature()

future_data_tbl <- data %>% 
  tk_index() %>% 
  tk_make_future_timeseries(n_future = 12, inspect_weekdays = TRUE, inspect_months = TRUE) %>% 
  tk_get_timeseries_signature()

# 4.2 MACHINE LEARNING ----

# XGBoost

seed <- 123
set.seed(seed)

model_xgboost <- boost_tree(
  mode = "regression",
  mtry = 20,
  trees = 500,
  min_n = 3, 
  tree_depth = 8,
  learn_rate = 0.01,
  loss_reduction = 0.01) %>% 
  set_engine(engine = "xgboost") %>% 
  fit.model_spec(total_sales ~., data = train_tbl %>% select(-date, -label_text, -diff))

# 4.3 MAKE PREDICTION & FORMAT OUTPUT ----

future_data_tbl

prediction_tbl <- predict(model_xgboost, new_data = future_data_tbl) %>% 
  bind_cols(future_data_tbl) %>% 
  select(.pred, index) %>% 
  rename(total_sales = .pred,
         date        = index) %>% 
  mutate(label_text  = str_glue("Date: {date}
                                Revenue: {scales::dollar(total_sales)}")) %>% 
  add_column(key = "Prediction")


output_tbl <- data %>% 
  add_column(key = "Actual") %>% 
  bind_rows(prediction_tbl)

output_tbl


# 4.4 FUNCTION ----

generate_forecast <- function(data, n_future = 12, seed = NULL){
  
  train_tbl <- data %>% 
    tk_augment_timeseries_signature()
  
  future_data_tbl <- data %>% 
    tk_index() %>% 
    tk_make_future_timeseries(n_future = n_future, inspect_weekdays = TRUE, inspect_months = TRUE) %>% 
    tk_get_timeseries_signature()
  
  time_scale <- data %>% 
    tk_index() %>% 
    tk_get_timeseries_summary() %>% 
    pull(scale)
  
  if (time_scale == "year") {
    
    model <- linear_reg(mode = "regression") %>% 
      set_engine(engine = "lm") %>% 
      fit.model_spec(total_sales ~ ., data = train_tbl %>% select(total_sales, index.num))
    
  } else {
    seed <- seed
    set.seed(seed)
    model <- boost_tree(
      mode = "regression",
      mtry = 20,
      trees = 500,
      min_n = 3, 
      tree_depth = 8,
      learn_rate = 0.01,
      loss_reduction = 0.01) %>% 
      set_engine(engine = "xgboost") %>% 
      fit.model_spec(total_sales ~., data = train_tbl %>% select(-date, -label_text, -diff))
  }
  
  prediction_tbl <- predict(model, new_data = future_data_tbl) %>% 
    bind_cols(future_data_tbl) %>% 
    select(.pred, index) %>% 
    rename(total_sales = .pred,
           date        = index) %>% 
    mutate(label_text  = str_glue("Date: {date}
                                Revenue: {scales::dollar(total_sales)}")) %>% 
    add_column(key = "Prediction")
  
  
  output_tbl <- data %>% 
    add_column(key = "Actual") %>% 
    bind_rows(prediction_tbl)
  
  return(output_tbl)
  
  
}


processed_data_tbl %>% 
  aggregate_time_series(time_unit = "month") %>% 
  generate_forecast(n_future = 12, seed = 123) %>% 
  tail(20)


# 5.0 PLOT FORECAST ----

# 5.1 PLOT ----

data <- processed_data_tbl %>% 
  aggregate_time_series(time_unit = "month") %>% 
  generate_forecast(n_future = 12, seed = 123)

g <- data %>% 
  ggplot(aes(date, total_sales, color = key)) +
  geom_line() +
  geom_point(aes(text = label_text), size = 0.01) + 
  geom_smooth(method = "loess", span = 0.2) +
  
  theme_tq() + 
  scale_color_tq() +
  scale_y_continuous(labels = scales::dollar_format()) +
  labs(x = "", y = "")

ggplotly(g, tooltip = "text")

# 5.2 FUNCTION ----

plot_forecast <- function(data) {

 # Yearly - LM Smoother
  
  time_scale <- data %>% 
    tk_index() %>% 
    tk_get_timeseries_summary() %>% 
    pull(scale)

  # Only 1 Prediction - points
  n_prediction <- data %>% 
    filter(key =="Prediction") %>% 
    nrow()
    
  
  g <- data %>% 
    ggplot(aes(date, total_sales, color = key)) +
    geom_line() +
  #  geom_point(aes(text = label_text), size = 0.01) + 
  #  geom_smooth(method = "loess", span = 0.2) +
    
    theme_tq() + 
    scale_color_tq() +
    scale_y_continuous(labels = scales::dollar_format()) +
    expand_limits(y = 0) +
    labs(x = "", y = "")
  
  # Yearly - LM Smoother
  if (time_scale == "year") {
    g <- g + 
      geom_smooth(method = "lm")
  } else {
    g <- g + geom_smooth(method = "loess", span = 0.2)
  }

  # Only 1 Prediction
  if (n_prediction == 1) {
    g <- g + geom_point(aes(text = label_text), size = 1)
  } else {
    g <- g + geom_point(aes(text = label_text), size = 0.01)
  }
  
  ggplotly(g, tooltip = "text")  
  
}


processed_data_tbl %>%
  aggregate_time_series(time_unit = "year") %>% 
  generate_forecast(n_future = 2, seed = 123) %>% 
  plot_forecast()



# 6.0 SAVE FUNCTIONS ----

dump(c("aggregate_time_series", "plot_time_series", "generate_forecast", "plot_forecast"),
     file = "Sales_Dashboard/00_scripts/04_demand_forecast.R")



