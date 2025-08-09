var file = null;

window.onload = function(){
	btnAbrirArchivo.onclick = function(){
		fupArchivo.click();
	}
	
	fupArchivo.onchange = function(event){
		file = this.files[0];
		txtArchivo.value = file.name;
		var reader = new FileReader();
		reader.onloadend = function(event){
			imgOriginal.src = reader.result;
		}
		reader.readAsDataURL(file);
	}
	
	btnClasificar.onclick = async function(){
		var token = document.getElementsByName("csrfmiddlewaretoken")[0].value;
		var frm = new FormData();
		var foto = imgOriginal.src.replace("data:image/png;base64,","").replace("data:image/jpeg;base64,","");
		frm.append("Foto", foto);
		frm.append("csrfmiddlewaretoken", token);
		var rptaHttp = await fetch("ClasificarObjeto", 
		{
			method: "POST",
			body: frm
		});
		if(rptaHttp.ok){
			var rptaTexto = await rptaHttp.text();
			if(rptaTexto!=""){
				spnRpta.innerText = rptaTexto;
			}
		}
	}
	
	btnNuevo.onclick = function(){		
		fupArchivo.value="";
		txtArchivo.value="";
		imgOriginal.src = "";
		spnRpta.innerText = "";
	}
}