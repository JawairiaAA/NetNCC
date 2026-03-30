from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cf
import matplotlib.pyplot as plt


def plot_maps_colorbar(ax, lon, lat, data, title,levels_custom, plot_label,cmap):
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,linewidth=1, color='lightgray', alpha=0.75, linestyle='-.')
    gl.top_labels = False
    gl.right_labels = False
    #gl.xlocator = mticker.FixedLocator([20, 25, 30, 35, 40])
    #gl.ylocator = mticker.FixedLocator([0, -5, -10, -15, -20, -25])
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 7, 'color': 'gray'}
    gl.ylabel_style = {'size': 7, 'color': 'gray'}
    # ax.set_title(title)
    # Countries
    m=ax.contourf(lon, lat, data, transform=ccrs.PlateCarree(), levels=levels_custom,cmap=cmap)  # this is the actual plot
    ax.coastlines()   ## adds coastlines
    ax.add_feature(cartopy.feature.BORDERS, linestyle='--',color='white'); # adds country borders
    cbar = plt.colorbar(m,fraction=0.03)  # adds colorbar
    cbar.set_label(plot_label)

def plot_maps_colorbar_lsta(ax, lon, lat, data, title,levels_custom, plot_label,cmap):
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,linewidth=1, color='black', alpha=0.75, linestyle='-.')
    gl.top_labels = False
    gl.right_labels = False
    #gl.xlocator = mticker.FixedLocator([20, 25, 30, 35, 40])
    #gl.ylocator = mticker.FixedLocator([0, -5, -10, -15, -20, -25])
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 7, 'color': 'gray'}
    gl.ylabel_style = {'size': 7, 'color': 'gray'}
    # ax.set_title(title)
    # Countries
    m=ax.contourf(lon, lat, data, transform=ccrs.PlateCarree(), levels=levels_custom,cmap=cmap)  # this is the actual plot
    ax.coastlines()   ## adds coastlines
    ax.add_feature(cartopy.feature.BORDERS, linestyle='--',color='white'); # adds country borders
    cbar = plt.colorbar(m,fraction=0.03)  # adds colorbar
    cbar.set_label(plot_label)
 