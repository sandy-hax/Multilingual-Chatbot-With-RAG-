/**
 * pacs.js
 * =======
 * Client-side script for PACS Operator Grievance Portal.
 */

document.addEventListener("DOMContentLoaded", () => {
    loadStats();
    loadGrievances();
});

async function loadStats() {
    try {
        const response = await fetch("/api/grievances/stats");
        const data = await response.json();
        document.getElementById("statTotal").innerText = data.total || 0;
        document.getElementById("statPending").innerText = data.pending || 0;
        document.getElementById("statInProgress").innerText = data.in_progress || 0;
        document.getElementById("statResolved").innerText = data.resolved || 0;
    } catch (err) {
        console.error("Failed to load stats:", err);
    }
}

async function loadGrievances() {
    const status = document.getElementById("filterStatus").value;
    const dept = document.getElementById("filterDept").value;

    const url = `/api/grievances?status=${encodeURIComponent(status)}&department=${encodeURIComponent(dept)}`;
    const tbody = document.getElementById("grievanceTableBody");

    try {
        const response = await fetch(url);
        const data = await response.json();

        if (!data.grievances || data.grievances.length === 0) {
            tbody.innerHTML = `
                <tr>
                    <td colspan="7" style="text-align: center; color: var(--text-muted); padding: 40px;">
                        No citizen grievances found for the selected filter.
                    </td>
                </tr>
            `;
            return;
        }

        tbody.innerHTML = "";
        data.grievances.forEach(g => {
            const tr = document.createElement("tr");
            const statusClass = (g.status || "Pending").replace(" ", "-");

            tr.innerHTML = `
                <td><strong style="color: var(--accent-indigo);">${escapeHtml(g.tracking_id)}</strong></td>
                <td>${escapeHtml(g.citizen_name)}</td>
                <td>${escapeHtml(g.phone_number)}</td>
                <td><span class="badge model-badge">${escapeHtml(g.department)}</span></td>
                <td>${escapeHtml(g.pacs_centre)} (${escapeHtml(g.district)})</td>
                <td><span class="badge-status ${statusClass}">${escapeHtml(g.status)}</span></td>
                <td>
                    <button class="btn btn-outline btn-sm" onclick="openTicketModal('${escapeHtml(g.tracking_id)}')">
                        <i class="fa-solid fa-pen-to-square"></i> Manage
                    </button>
                </td>
            `;
            tbody.appendChild(tr);
        });

        loadStats();
    } catch (err) {
        tbody.innerHTML = `
            <tr>
                <td colspan="7" style="text-align: center; color: var(--danger); padding: 30px;">
                    Failed to connect to Grievance API server.
                </td>
            </tr>
        `;
        console.error("Failed to load grievances:", err);
    }
}

async function openTicketModal(trackingId) {
    try {
        const response = await fetch(`/api/grievances/${encodeURIComponent(trackingId)}`);
        const g = await response.json();

        const modalBody = document.getElementById("ticketModalBody");
        modalBody.innerHTML = `
            <div style="display: flex; flex-direction: column; gap: 14px; font-size: 0.9rem;">
                <div style="display: flex; justify-content: space-between; border-bottom: 1px solid var(--card-border); padding-bottom: 10px;">
                    <div>
                        <strong style="font-size: 1.1rem; color: var(--accent-indigo);">${escapeHtml(g.tracking_id)}</strong>
                        <div style="color: var(--text-muted); font-size: 0.8rem;">Filed on ${escapeHtml(g.created_at || 'N/A')}</div>
                    </div>
                    <span class="badge-status ${(g.status || 'Pending').replace(' ', '-')}">${escapeHtml(g.status)}</span>
                </div>

                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
                    <div><strong>Citizen Name:</strong> ${escapeHtml(g.citizen_name)}</div>
                    <div><strong>Phone:</strong> ${escapeHtml(g.phone_number)}</div>
                    <div><strong>Department:</strong> ${escapeHtml(g.department)}</div>
                    <div><strong>PACS Centre / Location:</strong> ${escapeHtml(g.pacs_centre)}, ${escapeHtml(g.district)}</div>
                </div>

                <div>
                    <strong>Complaint Description:</strong>
                    <div style="background: var(--card-bg); padding: 10px; border-radius: var(--radius-sm); margin-top: 4px; color: var(--text-secondary);">
                        ${escapeHtml(g.description || 'N/A')}
                    </div>
                </div>

                <div>
                    <strong>Desired Resolution:</strong>
                    <div style="background: var(--card-bg); padding: 10px; border-radius: var(--radius-sm); margin-top: 4px; color: var(--text-secondary);">
                        ${escapeHtml(g.desired_resolution || 'N/A')}
                    </div>
                </div>

                <hr style="border-color: var(--card-border);">

                <div>
                    <label style="font-weight: 600; display: block; margin-bottom: 6px;">Update Status:</label>
                    <select id="modalStatusSelect" class="select-wrapper" style="width: 100%; padding: 8px;">
                        <option value="Pending" ${g.status === 'Pending' ? 'selected' : ''}>Pending</option>
                        <option value="In Progress" ${g.status === 'In Progress' ? 'selected' : ''}>In Progress</option>
                        <option value="Resolved" ${g.status === 'Resolved' ? 'selected' : ''}>Resolved</option>
                        <option value="Rejected" ${g.status === 'Rejected' ? 'selected' : ''}>Rejected</option>
                    </select>
                </div>

                <div>
                    <label style="font-weight: 600; display: block; margin-bottom: 6px;">PACS Resolution Remarks:</label>
                    <textarea id="modalRemarksText" rows="3" style="width: 100%; background: var(--card-bg); border: 1px solid var(--card-border); color: var(--text-primary); border-radius: var(--radius-sm); padding: 8px; outline: none;" placeholder="Enter official PACS action taken or resolution summary...">${escapeHtml(g.pacs_remarks || '')}</textarea>
                </div>

                <div style="display: flex; justify-content: flex-end; gap: 10px; margin-top: 10px;">
                    <button class="btn btn-secondary" onclick="closeTicketModal()">Cancel</button>
                    <button class="btn btn-primary" style="background: var(--accent-indigo); color: #fff;" onclick="saveTicketResolution('${escapeHtml(g.tracking_id)}')">
                        <i class="fa-solid fa-floppy-disk"></i> Save & Update
                    </button>
                </div>
            </div>
        `;

        document.getElementById("ticketModal").classList.remove("hidden");
    } catch (err) {
        alert("Failed to load ticket details.");
        console.error(err);
    }
}

async function saveTicketResolution(trackingId) {
    const newStatus = document.getElementById("modalStatusSelect").value;
    const newRemarks = document.getElementById("modalRemarksText").value.trim();

    try {
        const response = await fetch(`/api/grievances/${encodeURIComponent(trackingId)}`, {
            method: "PATCH",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                status: newStatus,
                pacs_remarks: newRemarks
            })
        });

        if (response.ok) {
            closeTicketModal();
            loadGrievances();
        } else {
            alert("Failed to update grievance ticket.");
        }
    } catch (err) {
        console.error("Save failed:", err);
        alert("Connection error saving ticket resolution.");
    }
}

function closeTicketModal() {
    document.getElementById("ticketModal").classList.add("hidden");
}

function escapeHtml(str) {
    if (!str) return "";
    return String(str).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}
