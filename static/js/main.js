document.addEventListener("DOMContentLoaded", () => {
    const dropZone    = document.getElementById("drop-zone");
    const fileInput   = document.getElementById("file-input");
    const folderInput = document.getElementById("folder-input");
    const btnFile     = document.getElementById("btn-file");
    const btnFolder   = document.getElementById("btn-folder");

    btnFile.addEventListener("click", () => fileInput.click());
    btnFolder.addEventListener("click", () => folderInput.click());
    fileInput.addEventListener("change", e => api.classifyFiles(Array.from(e.target.files)));
    folderInput.addEventListener("change", e => api.classifyFiles(Array.from(e.target.files)));

    dropZone.addEventListener("dragover", e => {
        e.preventDefault();
        dropZone.classList.add("dragover");
    });
    dropZone.addEventListener("dragleave", () => dropZone.classList.remove("dragover"));
    dropZone.addEventListener("drop", async e => {
        e.preventDefault();
        dropZone.classList.remove("dragover");
        const files = await api.getFilesFromDataTransfer(e.dataTransfer.items);
        api.classifyFiles(files);
    });

    api.fetchLatestBatch(60);
});
