library(plotly)

data <- read.table("E:/MuYang/Mustafa/ug/sobol/Plot5d/data.txt",head=T)

colors <- c('#4AC6B7', '#1972A4', '#965F8A')
m <- list(
  l = 50,
  r = 50,
  b = 100,
  t = 100,
  pad = 4
)

p <- plot_ly(data, x = ~a, y = ~b, z = ~Figure_of_merit, color = ~nb, size = ~r, colors = colors,
            marker = list(symbol = 'circle', sizemode = 'diameter'), sizes = c(1, 9),
			text = ~paste('alpha:', a, '<br>beta:', b, '<br>Figure of merit:', Figure_of_merit,
                           '<br>radius:', r)) %>%

	layout(autosize=F, width=800,height=800, margin =m,
	    title = 'Figures of merit vs. alpha beta',
        scene = list(xaxis = list(title = 'alpha(0~3)',
                      gridcolor = 'rgb(255, 255, 255)',
                      range = c(0.0, 3.0),
                      type = 'log',
                      zerolinewidth = 1,
                      ticklen = 5,
                      gridwidth = 2),
               yaxis = list(title = 'beta (0~1)',
                      gridcolor = 'rgb(255, 255, 255)',
                      range = c(0.0, 1.0),
                      zerolinewidth = 1,
                      ticklen = 5,
                      gridwith = 2),
               zaxis = list(title = 'Figure of merit',
                            gridcolor = 'rgb(255, 255, 255)',
                            type = 'log',
                            zerolinewidth = 1,
                            ticklen = 5,
                            gridwith = 2)),
         paper_bgcolor = 'rgb(243, 243, 243)',
         plot_bgcolor = 'rgb(243, 243, 243)')

chart_link = api_create(p, filename="plot5d_1")
chart_link
			 
#colors <- colors[as.numeric(data$nb)]


data <- read.table("E:/MuYang/Mustafa/ug/sobol/Plot5d/data.txt",head=T)

colors <- c('#4AC6B7', '#1972A4', '#965F8A')

p <- plot_ly(data, x = ~a, y = ~b, z = ~Figure_of_merit, color = ~nb, size = ~r, colors = colors,legendgroup = ~r,
            marker = list(symbol = 'circle', sizemode = 'diameter'), sizes = c(1, 9),
			text = ~paste('alpha:', a, '<br>beta:', b, '<br>Figure of merit:', Figure_of_merit,
                           '<br>radius:', r)) %>%

	layout(title = 'Figures of merit vs. alpha beta',
         scene = list(xaxis = list(title = 'alpha(0~3)',
                      gridcolor = 'rgb(255, 255, 255)',
                      range = c(0.0, 3.0)),
               yaxis = list(title = 'beta (0~1)',
                      gridcolor = 'rgb(255, 255, 255)',
                      range = c(0.0, 1.0)),
               zaxis = list(title = 'Figure of merit',
                            gridcolor = 'rgb(255, 255, 255)'),
		/*					
		  add_trace(type = "scatter",
				x = ~mpg, 
				y = ~disp,
				text = ~car,
				symbol = ~as.factor(cyl),
				mode = 'markers',  
				legendgroup="cyl",
				marker = list(color = "grey", size = 15)) %>%

     	  add_trace(type = "scatter",
					x = ~mpg, 
					y = ~disp,
					text = ~car,
					color = ~as.factor(gear),
					mode = 'markers', 
					legendgroup="gear") %>%

 		  add_annotations( text="Cylinders:", xref="paper", yref="paper",
						   x=1.02, xanchor="left",
						   y=0.9, yanchor="bottom",    # Same y as legend below
						   legendtitle=TRUE, showarrow=FALSE ) %>%

		  add_annotations( text="Gears:", xref="paper", yref="paper",
						   x=1.02, xanchor="left",
						   y=0.7, yanchor="bottom",    # Y depends on the height of the plot
						   legendtitle=TRUE, showarrow=FALSE ) %>%
		*/					
         paper_bgcolor = 'rgb(243, 243, 243)',
         plot_bgcolor = 'rgb(243, 243, 243)')