import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from ultralytics import YOLO
import pandas as pd
from shiny import App, render, ui, reactive

# ================= LOAD MODEL =================
try:
    MODEL_PATH = Path(__file__).parent / "best.pt"
except NameError:
    MODEL_PATH = Path("best.pt")

# If the web server doesn't find the weights file locally, it pulls it from Google Drive
if not MODEL_PATH.exists():
    print("Downloading model weights from Google Drive...")
    
    # TODO: REPLACE THIS ID WITH YOUR ACTUAL GOOGLE DRIVE FILE ID
    file_id = "1CQx8soSbJRVLeJwPASuiZmqMaN8Lfkyb"
    cloud_url = f"https://docs.google.com/uc?export=download&id={file_id}"
    
    try:
        # Set a user-agent header so Google Drive accepts the request from the server
        opener = urllib.request.build_opener()
        opener.addheaders = [('User-agent', 'Mozilla/5.0')]
        urllib.request.install_opener(opener)
        
        urllib.request.urlretrieve(cloud_url, MODEL_PATH)
        print("Download from Google Drive complete.")
    except Exception as e:
        print(f"Google Drive download failed: {e}")

try:
    model = YOLO(str(MODEL_PATH))
    print("Success: Loaded model weights perfectly.")
except Exception as e:
    print(f"Warning: Model could not initialize. Running in demo mode. Error: {e}")
    model = None
# JUPYTER VALIDATION HYPERPARAMETERS
CONF = 0.467
NMS_IOU = 0.70
IMG_SIZE = 640
MAX_DET = 3000

# CUSTOM AG-THEME STYLING CHIPS (Dark Slate Green & Accent Gold Accent)
custom_css = """
.navbar {
    background-color: #1b4332 !important; 
    padding-bottom: 15px !important;
}
.navbar-brand {
    color: #ffb703 !important;
    font-size: 28px !important;
    font-weight: 800 !important;
    letter-spacing: 1px;
    width: 100%;
    text-align: center;
    display: block !important;
    margin-bottom: 10px;
}
.nav-link {
    color: #f8f9fa !important;
    font-weight: 600 !important;
    border-radius: 4px;
    margin: 0 5px;
}
.nav-link.active {
    background-color: #ffb703 !important;
    color: #1b4332 !important;
}
.card-header {
    background-color: #2d6a4f !important;
    color: white !important;
    font-weight: bold;
}
.btn-success {
    background-color: #52b788 !important;
    border: none;
}
.btn-primary {
    background-color: #2d6a4f !important;
    border: none;
}
"""

# ================= USER INTERFACE (UI) =================
app_ui = ui.page_navbar(
    # --- TAB 1: BACKGROUND INFORMATION ---
    ui.nav_panel(
        "Background & Info",
        ui.layout_columns(
            ui.card(
                ui.card_header("About this Application"),
                ui.markdown("""
                ### AI-Driven Agronomy
                This application utilizes a custom-trained **YOLO (You Only Look Once)** computer vision object detection architecture 
                specifically tuned to locate and count individual corn kernels on an ear profile. 
                
                ### Why Sample Multiple Ears?
                Estimating entire field production from a single ear introduces massive sampling bias. By sampling between **5 to 10 representative ears** across a management zone, farmers can establish a statistically sound *average kernel cohort* per ear, lowering variability 
                and matching standard university yield component estimation frameworks.
                """),
            ),
            ui.card(
                ui.card_header("Yield Estimation Logic"),
                ui.markdown("""
                ### Standard Component Method
                The final calculations utilize the established agronomic formula scaled for regional metrics:
                
                1. **Emerged Plant Population:** Derived dynamically using your row-to-row and seed-to-seed spacing geometries over a standard hectare unit.
                2. **Kernels Per Bushel Adjustment:** Standard setups assume 90,000 kernels comprise a standard 56 lb bushel. This engine dynamically shrinks or expands that denominator based on your custom input seed test weight to mirror true kernel density variations.
                """)
            ),
            col_widths=[6, 6]
        )
    ),
    
    # --- TAB 2: INTERACTIVE SAMPLING ENGINE ---
    ui.nav_panel(
        "Interactive AI Sampling",
        ui.layout_columns(
            # Top Controls Row
            ui.card(
                ui.card_header("Configuration Panel"),
                ui.input_radio_buttons(
                    "view_type", 
                    "Select Profile View Option:", 
                    {"single": "Single Side View (Automatically multiplies count by 2)", 
                     "double": "Both Sides Photographed (Front & Back - No Multiplier)"},
                    inline=True
                ),
                ui.layout_columns(
                    ui.output_ui("ui_image_uploaders"),
                    ui.div(
                        ui.input_action_button("add_to_table_btn", "Verify & Log Ear into Sample", class_="btn-success w-100 mt-4 py-2 fw-bold text-white"),
                        ui.input_action_button("clear_all_btn", "Clear All Current Samples", class_="btn-danger w-100 mt-2 py-1")
                    ),
                    col_widths=[8, 4]
                )
            ),
            col_widths=[12]
        ),
        ui.layout_columns(
            # Main Interactive Plot Window
            ui.card(
                ui.card_header("Interactive AI Center-Dot Verification Workbench"),
                ui.output_plot("plot_interactive_dots", click={"id": "plot_click"}),
                ui.div(ui.output_text("txt_current_count_display"), style="font-weight: bold; font-size:16px; text-align: center; margin: 10px 0; color: #2d6a4f;"),
                ui.markdown("<p style='text-align:center; color: gray;'><b>Interactive Correction Mode:</b> If a kernel is missing, click on it to add a dot. Click a dot again to remove it.</p>")
            ),
            # Live Metrics Dashboard and Table Panel
            ui.div(
                ui.layout_columns(
                    ui.value_box("Total Ears Logged", ui.output_text("txt_total_ears"), theme="bg-gradient-blue-purple"),
                    ui.value_box("Cohort Average Kernels", ui.output_text("txt_avg_kernels"), theme="bg-gradient-orange-red"),
                    col_widths=[6, 6]
                ),
                ui.card(
                    ui.card_header("Logged Ear Data Table"),
                    ui.output_table("table_registry")
                ),
            ),
            col_widths=[7, 5]
        )
    ),
    
    # --- TAB 3: FIELD ESTIMATION ENGINE ---
    ui.nav_panel(
        "Estimate Field Yield",
        ui.layout_columns(
            ui.card(
                ui.card_header("Field Geometry & Crop Inputs"),
                ui.input_numeric("test_weight", "1. Test Weight (lbs/bu)", value=56.0, min=40.0, max=70.0, step=0.1),
                ui.input_numeric("row_spacing", "2. Row Spacing (inches)", value=30.0, min=10.0, max=40.0, step=0.5),
                ui.input_numeric("seed_spacing", "3. Seed Spacing (inches)", value=6.0, min=1.0, max=15.0, step=0.1),
                ui.input_numeric("total_hectares", "4. Total Field Area (Hectares)", value=10.0, min=0.1, step=0.5),
                ui.input_action_button("run_yield_btn", "Execute Field Production Modeling", class_="btn-primary w-100 mt-3 py-2 fw-bold")
            ),
            ui.card(
                ui.card_header("Final Yield Projections Output"),
                ui.markdown("### Production Analytics Dashboard"),
                ui.hr(),
                ui.output_ui("ui_yield_results_box")
            ),
            col_widths=[5, 7]
        )
    ),
    title="Precision Ag: AI Kernel Counter and Yield Estimator",
    header=ui.tags.style(custom_css)
)

# ================= SERVER LOGIC =================
def server(input, output, session):
    
    # SYSTEM STATES (Reactive values)
    sample_df = reactive.Value(pd.DataFrame(columns=["Ear ID", "Raw Count", "Type", "Total Ear Kernels"]))
    ear_index = reactive.Value(0)
    
    active_img_rgb = reactive.Value(None)
    interactive_dots = reactive.Value([])

    # 1. DYNAMIC INPUT CHIPS BASED ON SINGLE OR DOUBLE SELECTION
    @render.ui
    def ui_image_uploaders():
        if input.view_type() == "double":
            return ui.layout_columns(
                ui.input_file("img_side_a", "Upload Front Side Image", accept=[".jpg", ".jpeg", ".png"]),
                ui.input_file("img_side_b", "Upload Back Side Image", accept=[".jpg", ".jpeg", ".png"]),
                col_widths=[6, 6]
            )
        else:
            return ui.input_file("img_side_a", "Upload Single Side Ear Image", accept=[".jpg", ".jpeg", ".png"])

    # 2. RUN INFERENCE & MATRIX RENDERING
    @reactive.effect
    @reactive.event(input.img_side_a, input.img_side_b, input.view_type)
    def run_image_inference():
        file_a = input.img_side_a()
        if not file_a:
            return
            
        img_a = cv2.imread(file_a[0]["datapath"])
        if img_a is None:
            return
            
        found_dots = []
        
        # Process Side A
        if model is not None:
            res_a = model.predict(source=img_a, imgsz=IMG_SIZE, conf=CONF, iou=NMS_IOU, max_det=MAX_DET, verbose=False)[0]
            boxes_a = res_a.boxes.xyxy.cpu().numpy() if res_a.boxes is not None else []
            for box in boxes_a:
                cx = int((box[0] + box[2]) / 2)
                cy = int((box[1] + box[3]) / 2)
                found_dots.append((cx, cy))
        else:
            found_dots = [(100, 200), (150, 220), (200, 210), (250, 230)]
            
        # Process Side B if Double Mode is enabled
        file_b = input.img_side_b() if "img_side_b" in input else None
        if input.view_type() == "double" and file_b:
            img_b = cv2.imread(file_b[0]["datapath"])
            if img_b is not None:
                h_a, w_a, _ = img_a.shape
                img_b_resized = cv2.resize(img_b, (w_a, h_a))
                canvas = np.hstack((img_a, img_b_resized))
                
                if model is not None:
                    res_b = model.predict(source=img_b_resized, imgsz=IMG_SIZE, conf=CONF, iou=NMS_IOU, max_det=MAX_DET, verbose=False)[0]
                    boxes_b = res_b.boxes.xyxy.cpu().numpy() if res_b.boxes is not None else []
                    for box in boxes_b:
                        cx = int((box[0] + box[2]) / 2) + w_a
                        cy = int((box[1] + box[3]) / 2)
                        found_dots.append((cx, cy))
            else:
                canvas = img_a
        else:
            canvas = img_a
            
        canvas_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        active_img_rgb.set(canvas_rgb)
        interactive_dots.set(found_dots)

    # 3. INTERACTIVE PLOT CANVAS MOUSE LISTENER
    @reactive.effect
    @reactive.event(input.plot_click)
    def handle_canvas_click():
        click_data = input.plot_click()
        if click_data is None or active_img_rgb() is None:
            return
            
        clicked_x = int(click_data["x"])
        clicked_y = int(click_data["y"])
        
        current_points = list(interactive_dots.get())
        clicked_existing = False
        
        for pt in current_points:
            dist = np.sqrt((pt[0] - clicked_x)**2 + (pt[1] - clicked_y)**2)
            if dist < 18:  # 18-pixel proximity radius selection threshold
                current_points.remove(pt)
                clicked_existing = True
                break
                
        if not clicked_existing:
            current_points.append((clicked_x, clicked_y))
            
        interactive_dots.set(current_points)

    # 4. PLOT CANVAS CONTAINER GENERATOR
    @render.plot
    def plot_interactive_dots():
        base_canvas = active_img_rgb()
        fig, ax = plt.subplots(figsize=(12, 6))
        
        if base_canvas is not None:
            ax.imshow(base_canvas)
            pts = interactive_dots.get()
            for pt in pts:
                ax.plot(pt[0], pt[1], 'ro', markersize=5)
        else:
            ax.text(0.5, 0.5, "Upload your corn ear images above\nto open the interactive AI counting verification desk.", 
                    ha='center', va='center', fontsize=12, color='gray', weight='bold')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            
        ax.axis("off")
        plt.tight_layout()
        return fig

    @render.text
    def txt_current_count_display():
        if active_img_rgb() is None:
            return ""
        return f"Verified Bounding Anchors on Plot: {len(interactive_dots.get())} kernels."

    # 5. LOG DATA CHIPS INTO HISTORICAL SUMMARY RUN
    @reactive.effect
    @reactive.event(input.add_to_table_btn)
    def commit_ear_data():
        pts = interactive_dots.get()
        if not pts or active_img_rgb() is None:
            return
            
        raw_count = len(pts)
        mode = input.view_type()
        
        if mode == "single":
            calculated_total = raw_count * 2
            type_label = "1-Side (x2)"
        else:
            calculated_total = raw_count
            type_label = "2-Sides (x1)"
            
        next_id = ear_index.get() + 1
        ear_index.set(next_id)
        
        new_entry = pd.DataFrame([{
            "Ear ID": f"Ear #{next_id}",
            "Raw Count": raw_count,
            "Type": type_label,
            "Total Ear Kernels": calculated_total
        }])
        
        updated_df = pd.concat([sample_df.get(), new_entry], ignore_index=True)
        sample_df.set(updated_df)
        
        # Clear out uploader slate cleanly for subsequent uploads
        active_img_rgb.set(None)
        interactive_dots.set([])

    @reactive.effect
    @reactive.event(input.clear_all_btn)
    def clear_sample_registry():
        sample_df.set(pd.DataFrame(columns=["Ear ID", "Raw Count", "Type", "Total Ear Kernels"]))
        ear_index.set(0)
        active_img_rgb.set(None)
        interactive_dots.set([])

    # 6. CALCULATE INTERACTIVE METRIC CARDS
    @render.text
    def txt_total_ears():
        return f"{len(sample_df.get())} ears saved"

    @render.text
    def txt_avg_kernels():
        df = sample_df.get()
        if df.empty:
            return "N/A"
        return f"{round(df['Total Ear Kernels'].mean(), 1)} per ear"

    @render.table
    def table_registry():
        return sample_df.get()

    # 7. FIELD YIELD SIMULATOR CALCULATOR BLOCK
    @render.ui
    @reactive.event(input.run_yield_btn)
    def ui_yield_results_box():
        df = sample_df.get()
        if df.empty:
            return ui.markdown("<div class='alert alert-danger'><b>No Data Found:</b> Please add sampled ears in Tab 2 first!</div>")
            
        avg_kernels = df["Total Ear Kernels"].mean()
        sq_in_per_plant = input.row_spacing() * input.seed_spacing()
        
        if sq_in_per_plant <= 0:
            return ui.markdown("Invalid geometries.")
            
        plants_per_ha = 15500031.0 / sq_in_per_plant
        kernels_per_bu = 90000.0 * (56.0 / input.test_weight() if input.test_weight() > 0 else 1)
        
        bushels_per_ha = (plants_per_ha * avg_kernels) / kernels_per_bu
        total_production = bushels_per_ha * input.total_hectares()
        
        return ui.HTML(f"""
            <div style='padding: 18px; border-radius: 6px; background: #f8f9fa; border-left: 6px solid #ffb703;'>
                <p style='margin-bottom:5px;'><b>Calculated Hectare Baseline Population:</b> {int(plants_per_ha):,} plants/ha</p>
                <p style='margin-bottom:5px;'><b>Sample Pool Baseline Mean:</b> {round(avg_kernels, 1)} Total Kernels/Ear</p>
                <h3 style='color: #2d6a4f; margin-top:20px; font-weight:800;'>Estimated Productivity Projections:</h3>
                <h1 style='color: #2d6a4f;'>{round(bushels_per_ha, 1):,} <span style='font-size:18px; color:gray;'>bu / Ha</span></h1>
                <h3 style='color: #1b4332; margin-top:15px;'>Total Farm Production Volume Forecast ({input.total_hectares()} Ha):</h3>
                <h0 style='font-size: 42px; font-weight:900; color: #d9534f;'>{round(total_production, 1):,} <span style='font-size:22px; font-weight:normal; color:gray;'>Total Bushels</span></h0>
            </div>
        """)

# ================= EXPORT APP OBJECT =================
app = App(app_ui, server)
