#------------------- DASHBOARD STRUCTURE----------------------------------------

# Load packages ----------------------------------------------------------------
library(DT) # for data table
library(sf) # to read geoJSON files
library(readr) # to read csv files
library(dplyr) #for data manipulation
library(ggplot2) # for plots
library(leaflet) # for map
library(leaflet.extras) # for map
library(purrr) # for R to understand `%||%`
library(RColorBrewer) # for colors
library(magrittr) # for R to understand '%>%'
library(plotly) # for the plots and charts
library(shiny) # to run the shiny app
library(bslib) # for themes
library(shinythemes) # for themes
library(shinyWidgets) # slider, check boxes, etc

# 1. Load data -----------------------------------------------------------------

#Load the GeoJSON file containing country boundaries
outline <- st_read("D:\\Desktop\\510_Global\\Final submissions\\Final dashboard family\\Data\\geoBoundaries-CMR-ADM1.geojson")

#Load the .csv file containing the attribute information
incidence <- read.csv("D:\\Desktop\\510_Global\\Final submissions\\Final dashboard family\\Data\\Slider_Regional_Incidence.csv")

#For the time slider, it is important to make sure the date is in the right format
incidence$start_date <- as.Date(incidence$start_date, format="%d-%m-%Y")

#Obtaining the unique values of dates in the date range
date_range <- unique(incidence$start_date)

#Defining deaths per region
deaths <- count(incidence, ADM1_FR, wt = deaths)

#Listing the different regions in Cameroon
regions <- unique(incidence$ADM1_FR)

# Assuming start_date is a character vector with "DD-MM-YYYY" format
incidence <- mutate(incidence,
                    start_date = as.Date(start_date, format = "%d-%m-%Y"))

#Merging the .csv file and GeoJSON in the leaflet map, we need to merge data
merged <- merge(outline, incidence, by="ADM1_FR")

# 2. Define UI - User Interface ------------------------------------------------

ui <- 
  
  #Navbar talks about the various panels that would be visible in the application
  navbarPage(h4("Geovisualisation of Cholera in Cameroon"), collapsible = T, 
             
             #---------------- Cholera tab ---------- UI code --------------- 
             
             tabPanel(h5("Cholera Incidence Map"),
                      div(class="outer",
                          #Linking the CSS file for the styling of the dashboard
                          tags$head(includeCSS("styles.css")),
                          
                          leafletOutput("mymap", width="100%", height="100%"),
                          
                          absolutePanel(h3("Cholera Incidence Map"), 
                                        p("The map visualises the Cholera Incidence Cases in Cameroon over the period of time represented on the slider."), 
                                        p("The slider can be played to visualise the trend or be used to select the date the user requires information about:"),
                                        
                                        id = "controls", class = "panel panel-default",
                                        top = 100, left = 75, width = 400, fixed = T,
                                        draggable = F, height = "auto",
                                        
                                        sliderTextInput(inputId = "date_slider",
                                                        label = "Select Date:",
                                                        choices = format(unique(incidence$start_date), "%d %b %y"),
                                                        selected = format((min(incidence$start_date)), "%d %b %y"),
                                                        grid = F,
                                                        animate = animationOptions(interval = 3000, loop = FALSE)
                                        ),
                                        
                                        #adding the logos to the dashboard
                                        absolutePanel(class = "card",
                                                      bottom = 20, left = 60, fixed=TRUE, draggable = FALSE, height = "auto",
                                                      img(src='logo.png', height = 80, width = 320))
                                        
                          ))),
             
             #------------ Outbreak Plots tab ---------- UI code -------------------------
             
             tabPanel(h5("Outbreak Plots"), 
                      
                      tabsetPanel(
                        
                        tabPanel("Bar and Line Plots",
                                 
                                 h4("Region-wise total number of cases and deaths due to Cholera in Cameroon"),
                                 p("In this tab, the two plots have been created to understand the Cholera Outbreak in Cameroon:"),
                                 p("1. The first plot visualises the cumulative no. of cases in a bar chart and cumulative no. of deaths represented by a line: "),
                                 p("   The users can click on the region's bar in the first plot below to know more about the trend of cases and deaths in that particular region."),                                 
                                 p("2. The second plot visualises the no. of cases throughout the timeline in the Cameroon regions:"),
                                 p("   The user can hover on the respective region's line graph to know more about the history of Cholera cases in the region. This plot helps understand the trend of the Cholera cases in each region and also allow the user to compare it with the selected plot in the first plot."),
                                 
                                 #Plot - 1
                                 plotlyOutput("drilldown_plot", width = "900px", height = "300px"
                                 ),
                                 
                                 #To go to the previous page from the selected graph
                                 uiOutput("back_button"), 
                                 
                                 #Plot - 2
                                 plotlyOutput("time" , width = "900px", height = "300px")
                        ),
                        
                        tabPanel("Violin Plot",
                                 h3("Violin Plot"),
                                 p("The violin plot represents the distribution of cases in the regions in Cameroon. It is a combination of box plot and kernel density plot, that shows the peaks in the data. Violin plots give information about the statistical summary and density of the variable."),
                                 p("Hover on the plot to explore the data more and understand it."),
                                 fig
                        ))
             ),
             
             #---------------- Data tab ---------- UI code --------------------------------
             
             tabPanel(h5("Data"),
                      mainPanel(
                        h3("Cholera Incidence data in Cameroon"),
                        p(" In this panel, the raw data used to create the charts and plots has been displayed. Users can view, search, sort and download the data for further use or a better understanding"),
                        dataTableOutput(outputId = "Cdatatable"),
                        downloadButton("download_data", "Download data")
                      )
             ),
             
             #---------------- About tab ---------- UI code ------------------------------
             
             tabPanel(h5("About"),
                      h3("Source Code"), 
                      p("The Source Code for processing the dashboard is available from: ",
                        tags$a(href='https://github.com/rodekruis/RIPOSTE/tree/main/Shiny%20App%20Cholera', 'Shiny App Cholera')), #to add github link here
                      h3("Data Source"), 
                      div(
                        p("The administrational boundaries for Cameroon has been obtained from ",
                          tags$a(href='https://data.humdata.org/dataset/geoboundaries-admin-boundaries-for-cameroon?', 'Cameroon Geoboundaries')),
                        p("Data source for the visualisation is the CCOUSP | PHEOCC, See",
                          tags$a(href='https://www.ccousp.cm/urgences-sanitaires/cholera/situation-cholera-cameroun/', 'Cholera Situation Reports - Cameroon'),
                        ))
             ),
  )

# Defining breaks in Main tab of the dashboard ---------------------------------

#breaks for cases
breaks = c(0, 50, 100, 500, 1000, 1500)

#Adding color palette to the map, colorBin gets palette information from ColorBrewer2
pal <- colorBin("Reds", domain = merged$cases, bins = breaks)

basemap = leaflet(outline) %>%
  addTiles() %>%
  addProviderTiles(providers$CartoDB.Positron) %>%
  setView(12, 7, zoom = 5.5, options = list) %>%
  addLegend("bottomright",
            pal = pal,
            values = ~merged$cases,
            title="No. of Cases",
            opacity = 1
  )

# 3. Define server -------------------------------------------------------------

server <- function(input, output, session) {
  
  #---------------- Cholera tab ---------- server code -------------------------
  
  filtered_data <- reactive({
    #converting input date to date class
    input_date <- as.Date(input$date_slider, format = "%d %b %y")
    # Filter data based on input date
    filter(merged, start_date == input_date)
  })
  
  output$mymap <- renderLeaflet({
    basemap
  })
  
  # Rendering the leaflet map
  observeEvent(input$date_slider, {
    leafletProxy("mymap") %>%
      addTiles()   %>%
      addPolygons(data = filtered_data(),
                  fillColor = ~pal(filtered_data()$cases),
                  fillOpacity = 0.7,
                  color = "grey",
                  smoothFactor = 0.1,
                  stroke = TRUE,
                  weight = 1,
                  label= sprintf("Region: <b>%s</b><br>No. of Cases: %d<br/>No. of Deaths: %d", filtered_data()$ADM1_FR,filtered_data()$cases,filtered_data()$deaths) %>% lapply(htmltools::HTML),
                  highlightOptions = highlightOptions(color = "black", bringToFront = TRUE, weight = 2)
      )
  })  
  
  #------------ Outbreak Plots tab ---------- server code ---------------------- 
  
  # For Bar and Line Plot tab --------------------------
  # Define a custom color palette with 10 colors for each region - color blind friendly
  custom_palette <- c("#000000", "#2271B2", "#3DB7E9", "#8400CD","#9F0162", "#359B73", "#A40122", "#D55E00", "#E69F00", "#F0E442" )
  
  # Reactive values to track drill-down state
  current_region <- reactiveVal(NULL)

  # Filtered data based on drill-down
  plot_data <- reactive({
    if (is.null(current_region())) {
      #For Level 1 data filtering
      incidence %>%
        group_by(ADM1_FR) %>%
        summarize(cases = sum(cases), deaths = sum(deaths))
    } else {
      #Level 2 data filtering
      incidence %>%
        filter(ADM1_FR == current_region()) %>%
        arrange(start_date)
    }
  })
  
  # Plot 1 ----------------------------------------
  output$drilldown_plot <- renderPlotly({
    
    p <- if (is.null(current_region())) {
      # Level 1 Plot - regions on x-axis
      plot_ly(plot_data(),
              x = ~ADM1_FR, y = ~cases, type = 'bar', name = 'Cases') %>%
        add_trace(y = ~deaths, type = 'scatter', mode = 'lines+markers', name = 'Deaths', yaxis = 'y2')
    } else {
      # Level 2 Plot - start_date on x-axis
      plot_ly(plot_data(),
              x = ~start_date, y = ~cases, type = 'bar', name = 'Cases') %>%
        add_trace(y = ~deaths, type = 'scatter', mode = 'lines+markers', name = 'Deaths', yaxis = 'y2') %>%
        # Handle date summarization 
        layout(barmode = "group")  # Group bars are easier for comparison
      
    }
    
    #Shared Layout  
    p %>% 
      layout(
        xaxis = list(title = current_region() %||% "Regions in Cameroon"),
        yaxis = list(
          title = 'No. of cases',
          range = c(min(plot_data()$cases), 
                    max(plot_data()$cases) + 10)),
        yaxis2 = list(
          overlaying = 'y',
          title = 'No. of deaths',
          side = 'right',
          range = c(min(plot_data()$deaths),
                    max(plot_data()$deaths) + 10)),
        legend = list(title = list(text = '<b> Legend </b>'))
      ) %>% 
      # Control drill-down behavior
      event_register("plotly_click")  
  })
  
  # Drill-down logic
  observeEvent(event_data("plotly_click"), {
    current_region(event_data("plotly_click")$x)
  })
  
  #Plot - 2 ------------------------------------
  #Report cases overtime - all regions or a specified region if selected
  data_time <- reactive({
    incidence %>%
      count(ADM1_FR, start_date, wt = cases)
  })
  
  #Renders a Plotly plot of cases overtime
  output$time <- renderPlotly ({
    d <- setNames(data_time(), c("color", "x", "y"))
    
    plot_ly(d) %>%
      add_lines(x = ~x, y = ~y, type = 'scatter', mode = 'lines+markers', color= ~color, colors = custom_palette) %>%
      layout(xaxis = list(title = 'Time', 
                          range=c(min(date_range), max(date_range))),
             yaxis = list(title = 'No. of Cases'),
             legend = list(title = '<b> Legend </b>')
      ) 
  })
  
  # Back Button UI ---------------------------
  output$back_button <- renderUI({
    if (!is.null(current_region())) {
      actionButton("back", "Back to Regions")
    }
  })
  
  # Reset drill-down on back button click
  observeEvent(input$back, {
    current_region(NULL)
  })
  
  # For Violin Plot tab ----------------------
  fig <- incidence %>%
    plot_ly(
      x = ~ADM1_FR,
      y = ~cases,
      split = ~ADM1_FR,
      type = 'violin',
      #box = list(visible = T),  - can uncomment in case the user wants to see the combination of box and violin plot
      meanline = list (visible = T)
    ) 
  
  fig <- fig %>%
    layout(xaxis = list(title = "Regions"),
           yaxis = list(title = "Distribution of cases", 
                        zeroline = F)
    )
  
  #---------------- Data tab ---------- server code ----------------------------
  
  # Create reactive data frame
  variables_selected <- reactive({
    incidence %>% select(input$selected_var)
  })
  
  # Create data table
  output$Cdatatable <- DT::renderDataTable(
    datatable(
      data = incidence , #%>% select(input$selected_var),
      options = list(pageLength = 10),
      colnames = c('ID' = 1, 'Reporting Date'=2, 'Region Name'=3, 'No. of cases'=4,'No. of deaths'=5)
    )
  )
  
  # Download file
  output$download_data <- downloadHandler(
    filename = function() {
      paste0("Choleradata.csv")
    },
    content = function(file) {
      write_csv(incidence %>% select(input$selected_var), file)
    }
  )
}

# 4. Launching the Shiny app  --------------------------------------------------

shinyApp(ui, server)
