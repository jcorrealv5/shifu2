var video = document.getElementById('video');
var contexto = canvas.getContext("2d");

window.onload = async function(){
	try {
		const stream = await navigator.mediaDevices.getUserMedia({video: true, audio: false});
		video.srcObject = stream;
	}
	catch (error){
		console.log('Error:', error);
	}
	
	btnDetectar.onclick = async function(){
		await enviarFrame();
	}
}

async function enviarFrame(){
	contexto.drawImage(video, 0, 0, canvas.width, canvas.height);
	var data = canvas.toDataURL("image/png");
	var foto = data.replace("data:image/png;base64,","").replace("data:image/jpeg;base64,","");
	var token = document.getElementsByName("csrfmiddlewaretoken")[0].value;
	var frm = new FormData();
	frm.append("Foto", foto);
	frm.append("csrfmiddlewaretoken", token);
	var rptaHttp = await fetch("DetectarRostros", 
	{
		method: "POST",
		body: frm
	});
	if(rptaHttp.ok){
		var blob = await rptaHttp.blob();
		var nTotal = blob.size;
		if(nTotal>0){
			imgDetectada.src = URL.createObjectURL(blob);
			enviarFrame();
		}
	}
}